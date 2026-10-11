"""Model definitions with memory calculations based on the PDF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    """Configuration for a transformer model."""
    num_parameters: int = 30_000_000_000  # 30B parameters
    num_layers: int = 48
    hidden_size: int = 7168
    num_attention_heads: int = 56
    vocab_size: int = 128256
    max_seq_len: int = 4096

    # Precision settings
    weight_dtype: Literal["fp16", "bf16", "fp8"] = "fp16"
    optimizer_dtype: Literal["fp32"] = "fp32"

    @property
    def bytes_per_parameter(self) -> int:
        """Bytes per parameter for weights/gradients."""
        if self.weight_dtype in ("fp16", "bf16"):
            return 2
        elif self.weight_dtype == "fp8":
            return 1
        return 4

    @property
    def optimizer_bytes_per_parameter(self) -> int:
        """Bytes per parameter for optimizer state (always fp32)."""
        return 4


@dataclass
class MemoryBreakdown:
    """Memory breakdown for a model."""
    # Per parameter: weight(2) + grad(2) + fp32_weight(4) + 2*optimizer(8) = 16 bytes
    weight_gb: float
    gradient_gb: float
    fp32_weight_gb: float
    optimizer_state_gb: float
    total_gb: float

    @classmethod
    def from_config(cls, config: ModelConfig) -> MemoryBreakdown:
        params = config.num_parameters
        weight_bytes = config.bytes_per_parameter
        opt_bytes = config.optimizer_bytes_per_parameter

        weight_gb = (params * weight_bytes) / 1e9
        gradient_gb = (params * weight_bytes) / 1e9
        fp32_weight_gb = (params * 4) / 1e9  # fp32 copy
        optimizer_state_gb = (params * 2 * opt_bytes) / 1e9  # 2 running averages
        total_gb = weight_gb + gradient_gb + fp32_weight_gb + optimizer_state_gb

        return cls(
            weight_gb=weight_gb,
            gradient_gb=gradient_gb,
            fp32_weight_gb=fp32_weight_gb,
            optimizer_state_gb=optimizer_state_gb,
            total_gb=total_gb,
        )

    def to_dict(self) -> dict:
        return {
            "weight_gb": round(self.weight_gb, 2),
            "gradient_gb": round(self.gradient_gb, 2),
            "fp32_weight_gb": round(self.fp32_weight_gb, 2),
            "optimizer_state_gb": round(self.optimizer_state_gb, 2),
            "total_gb": round(self.total_gb, 2),
        }


@dataclass
class LayerMemory:
    """Memory for a single layer."""
    layer_idx: int
    weight_gb: float
    gradient_gb: float
    fp32_weight_gb: float
    optimizer_state_gb: float
    activation_gb: float = 0.0

    @property
    def total_gb(self) -> float:
        return (self.weight_gb + self.gradient_gb +
                self.fp32_weight_gb + self.optimizer_state_gb +
                self.activation_gb)


def create_model_config(
    num_parameters: int = 30_000_000_000,
    precision: Literal["fp16", "bf16", "fp8"] = "fp16",
) -> ModelConfig:
    """Create a model config with specified precision."""
    return ModelConfig(
        num_parameters=num_parameters,
        weight_dtype=precision,
    )


def estimate_activation_memory(
    config: ModelConfig,
    batch_size: int,
    seq_len: int,
) -> float:
    """Estimate activation memory in GB."""
    # Rough estimate: activations ~ 2 * batch * seq_len * hidden_size * num_layers * bytes
    bytes_per_element = 2 if config.weight_dtype in ("fp16", "bf16") else 1
    activations = (
        2 * batch_size * seq_len * config.hidden_size * config.num_layers * bytes_per_element
    )
    return activations / 1e9


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
