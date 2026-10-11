"""Virtual GPU cluster management."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .virtual_gpu import VirtualGPU


@dataclass
class ClusterStats:
    """Aggregated cluster statistics."""
    total_memory_gb: float
    total_allocated_gb: float
    total_peak_gb: float
    avg_utilization_pct: float
    total_flops: int
    total_compute_time_s: float
    gpu_stats: list[dict]


class VirtualGPUCluster:
    """Manages a cluster of virtual GPUs."""

    def __init__(
        self,
        world_size: int = 32,
        memory_per_gpu_gb: float = 74.5,
        compute_tflops: float = 100.0,
    ):
        self.world_size = world_size
        self.gpus: list[VirtualGPU] = [
            VirtualGPU(
                rank=i,
                memory_gb=memory_per_gpu_gb,
                compute_tflops=compute_tflops,
            )
            for i in range(world_size)
        ]
        self._lock = threading.Lock()
        self._communication_log: list[dict] = []

    def get_gpu(self, rank: int) -> VirtualGPU:
        return self.gpus[rank]

    def all_reduce(self, tensor_name: str, size_gb: float) -> float:
        """Simulate all-reduce across all GPUs. Returns communication time."""
        start = time.perf_counter()
        for gpu in self.gpus:
            if gpu.has_tensor(tensor_name):
                gpu.simulate_compute(int(size_gb * 1e9 * 2))
        comm_time = time.perf_counter() - start
        self._communication_log.append({
            "op": "all_reduce",
            "tensor": tensor_name,
            "size_gb": size_gb,
            "time_s": comm_time,
            "participants": self.world_size,
        })
        return comm_time

    def reduce_scatter(self, tensor_name: str, size_gb: float) -> float:
        """Simulate reduce-scatter. Each GPU gets 1/world_size of result."""
        start = time.perf_counter()
        slice_size = size_gb / self.world_size
        for gpu in self.gpus:
            if gpu.has_tensor(tensor_name):
                gpu.remove_tensor(tensor_name)
                gpu.store_tensor(f"{tensor_name}_slice", slice_size)
                gpu.simulate_compute(int(slice_size * 1e9 * 2))
        comm_time = time.perf_counter() - start
        self._communication_log.append({
            "op": "reduce_scatter",
            "tensor": tensor_name,
            "size_gb": size_gb,
            "slice_size_gb": slice_size,
            "time_s": comm_time,
        })
        return comm_time

    def all_gather(self, tensor_name: str, slice_size_gb: float) -> float:
        """Simulate all-gather. Each GPU contributes slice, all get full tensor."""
        start = time.perf_counter()
        full_size = slice_size_gb * self.world_size
        for gpu in self.gpus:
            if gpu.has_tensor(f"{tensor_name}_slice"):
                gpu.remove_tensor(f"{tensor_name}_slice")
                gpu.store_tensor(tensor_name, full_size)
                gpu.simulate_compute(int(full_size * 1e9))
        comm_time = time.perf_counter() - start
        self._communication_log.append({
            "op": "all_gather",
            "tensor": tensor_name,
            "size_gb": full_size,
            "slice_size_gb": slice_size_gb,
            "full_size_gb": full_size,
            "time_s": comm_time,
        })
        return comm_time

    def broadcast(self, tensor_name: str, size_gb: float, src_rank: int = 0) -> float:
        """Simulate broadcast from source rank to all."""
        start = time.perf_counter()
        src_gpu = self.gpus[src_rank]
        if src_gpu.has_tensor(tensor_name):
            for gpu in self.gpus:
                if gpu.rank != src_rank:
                    gpu.store_tensor(tensor_name, size_gb)
                    gpu.simulate_compute(int(size_gb * 1e9))
        comm_time = time.perf_counter() - start
        self._communication_log.append({
            "op": "broadcast",
            "tensor": tensor_name,
            "size_gb": size_gb,
            "src_rank": src_rank,
            "time_s": comm_time,
        })
        return comm_time

    def get_cluster_stats(self) -> ClusterStats:
        gpu_stats = [gpu.get_stats() for gpu in self.gpus]
        total_allocated = sum(s["memory_allocated_gb"] for s in gpu_stats)
        total_peak = sum(s["memory_peak_gb"] for s in gpu_stats)
        avg_util = sum(s["memory_utilization_pct"] for s in gpu_stats) / self.world_size
        total_flops = sum(s["compute_total_flops"] for s in gpu_stats)
        total_compute_time = sum(s["compute_time_s"] for s in gpu_stats)

        return ClusterStats(
            total_memory_gb=self.world_size * self.gpus[0].memory.total_memory_gb,
            total_allocated_gb=total_allocated,
            total_peak_gb=total_peak,
            avg_utilization_pct=avg_util,
            total_flops=total_flops,
            total_compute_time_s=total_compute_time,
            gpu_stats=gpu_stats,
        )

    def reset_all(self) -> None:
        for gpu in self.gpus:
            gpu.reset_stats()
        self._communication_log.clear()

    def get_communication_log(self) -> list[dict]:
        return self._communication_log.copy()

    def print_summary(self) -> None:
        stats = self.get_cluster_stats()
        print(f"\n{'='*60}")
        print(f"CLUSTER SUMMARY (World Size: {self.world_size})")
        print(f"{'='*60}")
        print(f"Total Memory:     {stats.total_memory_gb:.2f} GB")
        print(f"Allocated:        {stats.total_allocated_gb:.2f} GB")
        print(f"Peak Allocated:   {stats.total_peak_gb:.2f} GB")
        print(f"Avg Utilization:  {stats.avg_utilization_pct:.1f}%")
        print(f"Total FLOPs:      {stats.total_flops / 1e12:.2f} TFLOPs")
        print(f"Compute Time:     {stats.total_compute_time_s*1e6:.2f} µs")
        print("\nPer-GPU Breakdown:")
        for gpu_stat in stats.gpu_stats:
            print(f"  GPU {gpu_stat['rank']:2d}: {gpu_stat['memory_allocated_gb']:6.2f} GB "
                  f"(peak: {gpu_stat['memory_peak_gb']:6.2f} GB) "
                  f"| {gpu_stat['memory_utilization_pct']:5.1f}% "
                  f"| Tensors: {gpu_stat['tensors']}")
