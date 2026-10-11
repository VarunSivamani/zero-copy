#!/usr/bin/env python3
"""Main entry point for ZeRO simulation."""

from __future__ import annotations

import argparse
import json

from zero_sim.simulation.runner import (
    run_communication_analysis,
    run_memory_scaling_analysis,
    run_simulation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="ZeRO Distributed Training Simulation")
    parser.add_argument("--world-size", type=int, default=32, help="Number of GPUs")
    parser.add_argument("--model-params", type=int, default=30_000_000_000, help="Model parameters")
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp8"], default="fp16")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--memory-per-gpu", type=float, default=74.5)
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--scaling", action="store_true", help="Run memory scaling analysis")
    parser.add_argument("--comm", action="store_true", help="Run communication analysis")

    args = parser.parse_args()

    results = {}

    if args.scaling:
        results["scaling"] = run_memory_scaling_analysis(
            model_params=args.model_params,
            precision=args.precision,
            memory_per_gpu_gb=args.memory_per_gpu,
        )

    if args.comm:
        results["communication"] = run_communication_analysis(
            world_size=args.world_size,
            model_params=args.model_params,
            precision=args.precision,
        )

    if not args.scaling and not args.comm:
        results["simulation"] = run_simulation(
            world_size=args.world_size,
            model_params=args.model_params,
            precision=args.precision,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            memory_per_gpu_gb=args.memory_per_gpu,
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
