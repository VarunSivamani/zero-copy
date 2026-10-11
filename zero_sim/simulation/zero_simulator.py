"""ZeRO simulation implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from zero_sim.models.model import LayerMemory, MemoryBreakdown, ModelConfig
from zero_sim.virtual_gpu.cluster import VirtualGPUCluster


@dataclass
class ZeROStats:
    """Statistics for a ZeRO stage simulation."""
    stage: Literal["dp", "zero1", "zero2", "zero3"]
    world_size: int
    memory_per_gpu_gb: float
    peak_memory_per_gpu_gb: float
    total_communication_gb: float
    communication_time_s: float
    compute_time_s: float
    step_time_s: float
    memory_breakdown: dict
    fits_in_memory: bool


class ZeROSimulator(ABC):
    """Base class for ZeRO simulators."""

    def __init__(
        self,
        cluster: VirtualGPUCluster,
        config: ModelConfig,
        batch_size: int = 2,
        seq_len: int = 4096,
    ):
        self.cluster = cluster
        self.config = config
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.memory_breakdown = MemoryBreakdown.from_config(config)
        self.layer_memories = get_layer_memory_breakdown(config)

    @abstractmethod
    def setup(self) -> None:
        """Initialize model state on GPUs."""
        pass

    @abstractmethod
    def forward_pass(self) -> float:
        """Run forward pass. Returns time."""
        pass

    @abstractmethod
    def backward_pass(self) -> float:
        """Run backward pass. Returns time."""
        pass

    @abstractmethod
    def optimizer_step(self) -> float:
        """Run optimizer step. Returns time."""
        pass

    def run_step(self) -> ZeROStats:
        """Run a complete training step."""
        self.cluster.reset_all()
        self.setup()

        _ = self.cluster.get_cluster_stats()

        forward_time = self.forward_pass()
        backward_time = self.backward_pass()
        opt_time = self.optimizer_step()

        final_stats = self.cluster.get_cluster_stats()
        comm_log = self.cluster.get_communication_log()

        total_comm_gb = sum(entry["size_gb"] * entry.get("participants", 1)
                           for entry in comm_log)
        total_comm_time = sum(entry["time_s"] for entry in comm_log)

        peak_per_gpu = max(s["memory_peak_gb"] for s in final_stats.gpu_stats)
        fits = peak_per_gpu <= self.cluster.gpus[0].memory.total_memory_gb

        return ZeROStats(
            stage=self.stage_name,
            world_size=self.cluster.world_size,
            memory_per_gpu_gb=final_stats.total_allocated_gb / self.cluster.world_size,
            peak_memory_per_gpu_gb=peak_per_gpu,
            total_communication_gb=total_comm_gb,
            communication_time_s=total_comm_time,
            compute_time_s=forward_time + backward_time + opt_time,
            step_time_s=forward_time + backward_time + opt_time + total_comm_time,
            memory_breakdown=self._get_memory_breakdown_dict(),
            fits_in_memory=fits,
        )

    @property
    @abstractmethod
    def stage_name(self) -> Literal["dp", "zero1", "zero2", "zero3"]:
        pass

    def _get_memory_breakdown_dict(self) -> dict:
        return self.memory_breakdown.to_dict()


def get_layer_memory_breakdown(config: ModelConfig) -> list[LayerMemory]:
    """Get per-layer memory breakdown."""
    params_per_layer = config.num_parameters // config.num_layers
    weight_bytes = config.bytes_per_parameter
    opt_bytes = config.optimizer_bytes_per_parameter

    layers = []
    for i in range(config.num_layers):
        weight_gb = (params_per_layer * weight_bytes) / 1e9
        gradient_gb = (params_per_layer * weight_bytes) / 1e9
        fp32_weight_gb = (params_per_layer * 4) / 1e9
        optimizer_state_gb = (params_per_layer * 2 * opt_bytes) / 1e9

        layers.append(LayerMemory(
            layer_idx=i,
            weight_gb=weight_gb,
            gradient_gb=gradient_gb,
            fp32_weight_gb=fp32_weight_gb,
            optimizer_state_gb=optimizer_state_gb,
        ))
    return layers


class DataParallelSimulator(ZeROSimulator):
    """Data Parallelism (baseline) - every GPU has full model copy."""

    @property
    def stage_name(self) -> Literal["dp"]:
        return "dp"

    def setup(self) -> None:
        # Each GPU stores full model: weights, gradients, fp32 weights, optimizer state
        for gpu in self.cluster.gpus:
            # Model weights
            gpu.store_tensor("weights", self.memory_breakdown.weight_gb)
            # Gradients (same size as weights)
            gpu.store_tensor("gradients", self.memory_breakdown.gradient_gb)
            # FP32 master weights
            gpu.store_tensor("fp32_weights", self.memory_breakdown.fp32_weight_gb)
            # Optimizer state (2 running averages)
            gpu.store_tensor("optimizer_state", self.memory_breakdown.optimizer_state_gb)

    def forward_pass(self) -> float:
        # Forward pass computes activations
        activation_gb = self._estimate_activation_memory()
        for gpu in self.cluster.gpus:
            gpu.store_tensor("activations", activation_gb)
            # Simulate forward compute: ~2 * params * batch * seq FLOPs
            flops = 2 * self.config.num_parameters * self.batch_size * self.seq_len
            gpu.simulate_compute(flops)
        return 0.1  # Simulated time

    def backward_pass(self) -> float:
        # Backward computes gradients
        for gpu in self.cluster.gpus:
            # Simulate backward compute: ~2x forward
            flops = 4 * self.config.num_parameters * self.batch_size * self.seq_len
            gpu.simulate_compute(flops)
        # All-reduce gradients across all GPUs
        self.cluster.all_reduce("gradients", self.memory_breakdown.gradient_gb)
        return 0.15

    def optimizer_step(self) -> float:
        # Update weights using optimizer state
        for gpu in self.cluster.gpus:
            flops = self.config.num_parameters * 10  # Rough optimizer ops
            gpu.simulate_compute(flops)
        return 0.05

    def _estimate_activation_memory(self) -> float:
        from zero_sim.models.model import estimate_activation_memory
        return estimate_activation_memory(self.config, self.batch_size, self.seq_len)


class ZeRO1Simulator(ZeROSimulator):
    """ZeRO Stage 1: Partition optimizer state (fp32 weights + 2 running averages)."""

    @property
    def stage_name(self) -> Literal["zero1"]:
        return "zero1"

    def setup(self) -> None:
        slice_factor = self.cluster.world_size
        # Optimizer state = fp32_weights (4 bytes) + 2 running averages (8 bytes) = 12 bytes
        optimizer_total_gb = self.memory_breakdown.fp32_weight_gb + self.memory_breakdown.optimizer_state_gb
        opt_slice = optimizer_total_gb / slice_factor

        for gpu in self.cluster.gpus:
            # Full model weights (replicated) - 2 bytes
            gpu.store_tensor("weights", self.memory_breakdown.weight_gb)
            # Full gradients (replicated) - 2 bytes
            gpu.store_tensor("gradients", self.memory_breakdown.gradient_gb)
            # Only 1/slice of optimizer state (fp32 weights + 2 running averages) - 12/8 = 1.5 bytes
            gpu.store_tensor("optimizer_state", opt_slice)

    def forward_pass(self) -> float:
        activation_gb = self._estimate_activation_memory()
        for gpu in self.cluster.gpus:
            gpu.store_tensor("activations", activation_gb)
            flops = 2 * self.config.num_parameters * self.batch_size * self.seq_len
            gpu.simulate_compute(flops)
        return 0.1

    def backward_pass(self) -> float:
        for gpu in self.cluster.gpus:
            flops = 4 * self.config.num_parameters * self.batch_size * self.seq_len
            gpu.simulate_compute(flops)

        # Reduce-scatter gradients
        self.cluster.reduce_scatter("gradients", self.memory_breakdown.gradient_gb)
        # All-gather updated weights (to keep replicas in sync)
        slice_size = self.memory_breakdown.gradient_gb / self.cluster.world_size
        self.cluster.all_gather("gradients", slice_size)
        return 0.15

    def optimizer_step(self) -> float:
        # Each GPU updates only its slice of optimizer state
        for gpu in self.cluster.gpus:
            optimizer_total_gb = self.memory_breakdown.fp32_weight_gb + self.memory_breakdown.optimizer_state_gb
            opt_slice = optimizer_total_gb / self.cluster.world_size
            flops = int(opt_slice * 1e9 / 4 * 10)
            gpu.simulate_compute(flops)

        # All-gather updated weights
        weight_slice = self.memory_breakdown.weight_gb / self.cluster.world_size
        self.cluster.all_gather("weights", weight_slice)
        return 0.05

    def _estimate_activation_memory(self) -> float:
        from zero_sim.models.model import estimate_activation_memory
        return estimate_activation_memory(self.config, self.batch_size, self.seq_len)


class ZeRO2Simulator(ZeROSimulator):
    """ZeRO Stage 2: Partition optimizer state + gradients."""

    @property
    def stage_name(self) -> Literal["zero2"]:
        return "zero2"

    def setup(self) -> None:
        slice_factor = self.cluster.world_size
        # Optimizer state = fp32_weights (4 bytes) + 2 running averages (8 bytes) = 12 bytes
        optimizer_total_gb = self.memory_breakdown.fp32_weight_gb + self.memory_breakdown.optimizer_state_gb
        opt_slice = optimizer_total_gb / slice_factor
        grad_slice = self.memory_breakdown.gradient_gb / slice_factor

        for gpu in self.cluster.gpus:
            # Full model weights (replicated) - 2 bytes
            gpu.store_tensor("weights", self.memory_breakdown.weight_gb)
            # Only 1/slice of gradients - 2/8 = 0.25 bytes
            gpu.store_tensor("gradients", grad_slice)
            # Only 1/slice of optimizer state (fp32 weights + 2 running averages) - 12/8 = 1.5 bytes
            gpu.store_tensor("optimizer_state", opt_slice)

    def forward_pass(self) -> float:
        activation_gb = self._estimate_activation_memory()
        for gpu in self.cluster.gpus:
            gpu.store_tensor("activations", activation_gb)
            flops = 2 * self.config.num_parameters * self.batch_size * self.seq_len
            gpu.simulate_compute(flops)
        return 0.1

    def backward_pass(self) -> float:
        for gpu in self.cluster.gpus:
            flops = 4 * self.config.num_parameters * self.batch_size * self.seq_len
            gpu.simulate_compute(flops)

        # Reduce-scatter gradients (gradients already partitioned)
        self.cluster.reduce_scatter("gradients", self.memory_breakdown.gradient_gb)
        # All-gather weights for next step
        weight_slice = self.memory_breakdown.weight_gb / self.cluster.world_size
        self.cluster.all_gather("weights", weight_slice)
        return 0.15

    def optimizer_step(self) -> float:
        # Each GPU updates only its slice
        for gpu in self.cluster.gpus:
            optimizer_total_gb = self.memory_breakdown.fp32_weight_gb + self.memory_breakdown.optimizer_state_gb
            opt_slice = optimizer_total_gb / self.cluster.world_size
            flops = int(opt_slice * 1e9 / 4 * 10)
            gpu.simulate_compute(flops)

        # All-gather updated weights
        weight_slice = self.memory_breakdown.weight_gb / self.cluster.world_size
        self.cluster.all_gather("weights", weight_slice)
        return 0.05

    def _estimate_activation_memory(self) -> float:
        from zero_sim.models.model import estimate_activation_memory
        return estimate_activation_memory(self.config, self.batch_size, self.seq_len)


class ZeRO3Simulator(ZeROSimulator):
    """ZeRO Stage 3: Partition weights + optimizer state + gradients."""

    @property
    def stage_name(self) -> Literal["zero3"]:
        return "zero3"

    def setup(self) -> None:
        slice_factor = self.cluster.world_size
        weight_slice = self.memory_breakdown.weight_gb / slice_factor
        grad_slice = self.memory_breakdown.gradient_gb / slice_factor
        fp32_slice = self.memory_breakdown.fp32_weight_gb / slice_factor
        opt_slice = self.memory_breakdown.optimizer_state_gb / slice_factor

        for gpu in self.cluster.gpus:
            # Only 1/slice of weights
            gpu.store_tensor("weights", weight_slice)
            # Only 1/slice of gradients
            gpu.store_tensor("gradients", grad_slice)
            # Only 1/slice of FP32 weights
            gpu.store_tensor("fp32_weights", fp32_slice)
            # Only 1/slice of optimizer state
            gpu.store_tensor("optimizer_state", opt_slice)

    def forward_pass(self) -> float:
        activation_gb = self._estimate_activation_memory()
        total_time = 0.0

        # For each layer, all-gather weights, compute, then free
        for layer in self.layer_memories:
            layer_weight_gb = layer.weight_gb

            # All-gather this layer's weights
            slice_size = layer_weight_gb / self.cluster.world_size
            self.cluster.all_gather(f"layer_{layer.layer_idx}_weights", slice_size)

            for gpu in self.cluster.gpus:
                gpu.store_tensor("activations", activation_gb / self.config.num_layers)
                params_per_layer = self.config.num_parameters // self.config.num_layers
                flops = 2 * params_per_layer * self.batch_size * self.seq_len
                gpu.simulate_compute(flops)

                # Free weights after use
                gpu.remove_tensor(f"layer_{layer.layer_idx}_weights")

            total_time += 0.1 / self.config.num_layers

        return total_time

    def backward_pass(self) -> float:
        total_time = 0.0

        # Backward pass in reverse layer order
        for layer in reversed(self.layer_memories):
            layer_weight_gb = layer.weight_gb
            slice_size = layer_weight_gb / self.cluster.world_size

            # All-gather weights for backward
            self.cluster.all_gather(f"layer_{layer.layer_idx}_weights", slice_size)

            for gpu in self.cluster.gpus:
                params_per_layer = self.config.num_parameters // self.config.num_layers
                flops = 4 * params_per_layer * self.batch_size * self.seq_len
                gpu.simulate_compute(flops)

                # Free weights
                gpu.remove_tensor(f"layer_{layer.layer_idx}_weights")

            # Reduce-scatter gradients for this layer
            self.cluster.reduce_scatter(f"layer_{layer.layer_idx}_gradients", layer.gradient_gb)

            total_time += 0.15 / self.config.num_layers

        return total_time

    def optimizer_step(self) -> float:
        # Each GPU updates its slice
        for gpu in self.cluster.gpus:
            opt_slice = self.memory_breakdown.optimizer_state_gb / self.cluster.world_size
            flops = int(opt_slice * 1e9 / 4 * 10)
            gpu.simulate_compute(flops)
        return 0.05

    def _estimate_activation_memory(self) -> float:
        from zero_sim.models.model import estimate_activation_memory
        return estimate_activation_memory(self.config, self.batch_size, self.seq_len)
