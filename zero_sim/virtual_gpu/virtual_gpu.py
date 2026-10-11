"""Virtual GPU simulation with memory and compute tracking."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryTracker:
    """Tracks memory allocation on a virtual GPU."""
    total_memory_gb: float
    allocated_gb: float = 0.0
    peak_allocated_gb: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def allocate(self, gb: float) -> bool:
        """Allocate memory. Returns True if successful."""
        with self._lock:
            if self.allocated_gb + gb <= self.total_memory_gb:
                self.allocated_gb += gb
                self.peak_allocated_gb = max(self.peak_allocated_gb, self.allocated_gb)
                return True
            return False

    def free(self, gb: float) -> None:
        """Free allocated memory."""
        with self._lock:
            self.allocated_gb = max(0.0, self.allocated_gb - gb)

    @property
    def free_gb(self) -> float:
        return self.total_memory_gb - self.allocated_gb

    @property
    def utilization_pct(self) -> float:
        return (self.allocated_gb / self.total_memory_gb) * 100


@dataclass
class ComputeTracker:
    """Tracks compute operations on a virtual GPU."""
    total_flops: int = 0
    total_time_s: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_compute(self, flops: int, time_s: float) -> None:
        with self._lock:
            self.total_flops += flops
            self.total_time_s += time_s

    @property
    def gflops(self) -> float:
        if self.total_time_s == 0:
            return 0.0
        return (self.total_flops / 1e9) / self.total_time_s


class VirtualGPU:
    """Simulates a single GPU with memory and compute tracking."""

    def __init__(
        self,
        rank: int,
        memory_gb: float = 74.5,
        compute_tflops: float = 100.0,
    ):
        self.rank = rank
        self.memory = MemoryTracker(total_memory_gb=memory_gb)
        self.compute = ComputeTracker()
        self.compute_tflops = compute_tflops
        self._data: dict[str, Any] = {}
        self._lock = threading.Lock()

    def store_tensor(self, name: str, size_gb: float, data: Any = None) -> bool:
        """Store a tensor on this GPU."""
        # For simulation, always track the allocation even if it exceeds physical memory
        # This allows us to see theoretical memory requirements
        self.memory.allocated_gb += size_gb
        self.memory.peak_allocated_gb = max(self.memory.peak_allocated_gb, self.memory.allocated_gb)
        with self._lock:
            self._data[name] = {"size_gb": size_gb, "data": data}
        return True

    def remove_tensor(self, name: str) -> float:
        """Remove a tensor and return freed memory."""
        with self._lock:
            if name in self._data:
                size_gb = self._data[name]["size_gb"]
                del self._data[name]
                self.memory.free(size_gb)
                return size_gb
        return 0.0

    def get_tensor_size(self, name: str) -> float:
        with self._lock:
            return self._data.get(name, {}).get("size_gb", 0.0)

    def has_tensor(self, name: str) -> bool:
        with self._lock:
            return name in self._data

    def simulate_compute(self, flops: int) -> float:
        """Simulate compute operation, return time taken."""
        time_s = flops / (self.compute_tflops * 1e12)
        self.compute.add_compute(flops, time_s)
        time.sleep(time_s * 1e-6)
        return time_s

    def get_stats(self) -> dict:
        return {
            "rank": self.rank,
            "memory_allocated_gb": self.memory.allocated_gb,
            "memory_peak_gb": self.memory.peak_allocated_gb,
            "memory_free_gb": self.memory.free_gb,
            "memory_utilization_pct": self.memory.utilization_pct,
            "compute_gflops": self.compute.gflops,
            "compute_total_flops": self.compute.total_flops,
            "compute_time_s": self.compute.total_time_s,
            "tensors": list(self._data.keys()),
        }

    def reset_stats(self) -> None:
        self.memory.allocated_gb = 0.0
        self.memory.peak_allocated_gb = 0.0
        self.compute.total_flops = 0
        self.compute.total_time_s = 0.0
        with self._lock:
            self._data.clear()
