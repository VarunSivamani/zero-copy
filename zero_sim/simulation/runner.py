"""Main simulation runner."""

from __future__ import annotations

from dataclasses import asdict
from typing import Literal

from zero_sim.models.model import MemoryBreakdown, create_model_config
from zero_sim.simulation.zero_simulator import (
    DataParallelSimulator,
    ZeRO1Simulator,
    ZeRO2Simulator,
    ZeRO3Simulator,
)
from zero_sim.virtual_gpu.cluster import VirtualGPUCluster


def run_simulation(
    world_size: int = 32,
    model_params: int = 30_000_000_000,
    precision: Literal["fp16", "bf16", "fp8"] = "fp16",
    batch_size: int = 2,
    seq_len: int = 4096,
    memory_per_gpu_gb: float = 74.5,
) -> dict:
    """Run all ZeRO stages and return comparison."""
    config = create_model_config(model_params, precision)
    cluster = VirtualGPUCluster(
        world_size=world_size,
        memory_per_gpu_gb=memory_per_gpu_gb,
    )

    print(f"\n{'='*70}")
    print(f"ZeRO SIMULATION: {model_params/1e9:.0f}B Model, {world_size} GPUs")
    print(f"GPU Memory: {memory_per_gpu_gb} GB, Precision: {precision}")
    print(f"{'='*70}")

    # Print theoretical memory breakdown
    mem_breakdown = MemoryBreakdown.from_config(config)
    print("\nTheoretical Memory (per GPU, full model):")
    for k, v in mem_breakdown.to_dict().items():
        print(f"  {k}: {v:.2f} GB")
    print(f"  Total: {mem_breakdown.total_gb:.2f} GB")

    simulators = [
        DataParallelSimulator(cluster, config, batch_size, seq_len),
        ZeRO1Simulator(cluster, config, batch_size, seq_len),
        ZeRO2Simulator(cluster, config, batch_size, seq_len),
        ZeRO3Simulator(cluster, config, batch_size, seq_len),
    ]

    results = {}
    for sim in simulators:
        print(f"\n--- Running {sim.stage_name.upper()} ---")
        stats = sim.run_step()
        results[sim.stage_name] = asdict(stats)

        print(f"  Memory/GPU:      {stats.memory_per_gpu_gb:.2f} GB")
        print(f"  Peak Memory/GPU: {stats.peak_memory_per_gpu_gb:.2f} GB")
        print(f"  Fits in GPU:     {'YES' if stats.fits_in_memory else 'NO'}")
        print(f"  Comm Volume:     {stats.total_communication_gb:.2f} GB")
        print(f"  Comm Time:       {stats.communication_time_s*1000:.2f} ms")
        print(f"  Compute Time:    {stats.compute_time_s*1000:.2f} ms")
        print(f"  Step Time:       {stats.step_time_s*1000:.2f} ms")

    return results


def run_memory_scaling_analysis(
    model_params: int = 30_000_000_000,
    precision: Literal["fp16", "bf16", "fp8"] = "fp16",
    memory_per_gpu_gb: float = 74.5,
) -> dict:
    """Analyze memory scaling across world sizes."""
    config = create_model_config(model_params, precision)
    world_sizes = [8, 16, 32, 64, 128, 256]

    print(f"\n{'='*70}")
    print(f"MEMORY SCALING ANALYSIS: {model_params/1e9:.0f}B Model")
    print(f"{'='*70}")

    results = {}
    for ws in world_sizes:
        cluster = VirtualGPUCluster(
            world_size=ws,
            memory_per_gpu_gb=memory_per_gpu_gb,
        )

        ws_results = {}
        for sim_class in [DataParallelSimulator, ZeRO1Simulator, ZeRO2Simulator, ZeRO3Simulator]:
            sim = sim_class(cluster, config, batch_size=2, seq_len=4096)
            cluster.reset_all()
            sim.setup()
            stats = cluster.get_cluster_stats()
            peak_per_gpu = max(s["memory_peak_gb"] for s in stats.gpu_stats)
            fits = peak_per_gpu <= memory_per_gpu_gb

            ws_results[sim.stage_name] = {
                "peak_memory_gb": peak_per_gpu,
                "fits": fits,
            }

        results[ws] = ws_results

        # Print summary row
        dp_fit = "✓" if ws_results["dp"]["fits"] else "✗"
        z1_fit = "✓" if ws_results["zero1"]["fits"] else "✗"
        z2_fit = "✓" if ws_results["zero2"]["fits"] else "✗"
        z3_fit = "✓" if ws_results["zero3"]["fits"] else "✗"
        print(f"  World Size {ws:3d}: DP={ws_results['dp']['peak_memory_gb']:6.1f}GB({dp_fit}) "
              f"Z1={ws_results['zero1']['peak_memory_gb']:6.1f}GB({z1_fit}) "
              f"Z2={ws_results['zero2']['peak_memory_gb']:6.1f}GB({z2_fit}) "
              f"Z3={ws_results['zero3']['peak_memory_gb']:6.1f}GB({z3_fit})")

    return results


def run_communication_analysis(
    world_size: int = 32,
    model_params: int = 30_000_000_000,
    precision: Literal["fp16", "bf16", "fp8"] = "fp16",
) -> dict:
    """Analyze communication volume for each stage."""
    config = create_model_config(model_params, precision)

    p_gb = config.num_parameters * config.bytes_per_parameter / 1e9  # GB

    print(f"\n{'='*70}")
    print(f"COMMUNICATION ANALYSIS: P = {p_gb:.1f} GB")
    print(f"{'='*70}")

    stages: dict[str, dict[str, int | str]] = {
        "Data Parallel": {
            "comm_multiplier": 2,
            "desc": "All-reduce gradients (2P)",
        },
        "ZeRO-1": {
            "comm_multiplier": 2,
            "desc": "Reduce-scatter + All-gather optimizer (2P)",
        },
        "ZeRO-2": {
            "comm_multiplier": 2,
            "desc": "Reduce-scatter grads + All-gather weights (2P)",
        },
        "ZeRO-3": {
            "comm_multiplier": 3,
            "desc": (
                "Forward all-gather + Backward all-gather + reduce-scatter (3P)"
            ),
        },
    }

    results = {}
    for name, info in stages.items():
        comm_gb = float(info["comm_multiplier"]) * p_gb
        # NVLink: ~450 GB/s, InfiniBand: ~50 GB/s
        nvlink_time = comm_gb / 450
        ib_time = comm_gb / 50

        results[name] = {
            "comm_gb": comm_gb,
            "multiplier": info["comm_multiplier"],
            "nvlink_time_s": nvlink_time,
            "ib_time_s": ib_time,
            "description": info["desc"],
        }

        print(f"  {name:15s}: {info['comm_multiplier']}P = {comm_gb:6.1f} GB "
              f"| NVLink: {nvlink_time*1000:6.1f} ms | InfiniBand: {ib_time*1000:6.1f} ms")

    return results
