# ZeRO Distributed Training Simulation

A Python simulation of Data Parallelism and ZeRO (Zero Redundancy Optimizer) stages 1-3 for distributed training of large language models. This project implements the concepts from **Session 12: Distributed Training I, Data Parallel and ZeRO**.

## Overview

This simulation demonstrates how different distributed training strategies affect memory usage and communication overhead when training a 30B parameter model across multiple GPUs.

### Key Concepts Implemented

| Stage | What's Partitioned | Bytes/Param | Memory at 8 GPUs | Fits in 74.5 GB? |
|-------|-------------------|-------------|------------------|------------------|
| Data Parallel | Nothing (full replica) | 16 | 447 GB | ❌ |
| ZeRO-1 | Optimizer state | 5.5 | 154 GB | ❌ |
| ZeRO-2 | Optimizer + Gradients | 3.75 | 105 GB | ❌ (fits at 32+) |
| ZeRO-3 | Weights + Optimizer + Gradients | 2.0 | 56 GB | ✅ |

**Communication Cost:**
- Data Parallel / ZeRO-1 / ZeRO-2: **2P** per step (all-reduce or reduce-scatter + all-gather)
- ZeRO-3: **3P** per step (extra all-gathers in forward/backward)

Where P = 60 GB for a 30B model in fp16.

## Project Structure

```
├── main.py                          # CLI entry point
├── pyproject.toml                   # Project configuration
├── zero_sim/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── model.py                 # Model config & memory calculations
│   ├── virtual_gpu/
│   │   ├── __init__.py
│   │   ├── virtual_gpu.py           # Single GPU simulation
│   │   └── cluster.py               # Multi-GPU cluster management
│   └── simulation/
│       ├── __init__.py
│       ├── zero_simulator.py        # ZeRO stage implementations
│       └── runner.py                # High-level simulation runners
└── notebooks/
    └── zero_simulation_demo.ipynb   # Interactive demonstration
```

## Installation

```bash
cd repo
pip install -e .
```

Or install dependencies directly:
```bash
pip install pandas matplotlib seaborn jupyter
```

## Usage

### Command Line

Run a single simulation (32 GPUs, 30B model):
```bash
python main.py --world-size 32 --model-params 30000000000
```

Run memory scaling analysis:
```bash
python main.py --scaling --model-params 30000000000
```

Run communication analysis:
```bash
python main.py --comm --world-size 32 --model-params 30000000000
```

Save results to JSON:
```bash
python main.py --world-size 32 --output results.json
```

### Jupyter Notebook

```bash
cd repo
jupyter notebook notebooks/zero_simulation_demo.ipynb
```

### Python API

```python
from zero_sim.simulation.runner import run_simulation, run_memory_scaling_analysis

# Run simulation
results = run_simulation(
    world_size=32,
    model_params=30_000_000_000,
    precision="fp16",
    batch_size=2,
    seq_len=4096,
    memory_per_gpu_gb=74.5,
)

# Run scaling analysis
scaling = run_memory_scaling_analysis(
    model_params=30_000_000_000,
    precision="fp16",
)
```

## Simulation Details

### Virtual GPU Model

Each virtual GPU tracks:
- **Memory**: Total, allocated, peak, and free memory (GB)
- **Compute**: Total FLOPs, compute time, effective GFLOPS
- **Tensors**: Named tensors with sizes stored on the GPU

### Cluster Operations

The cluster simulates collective operations:
- **All-reduce**: All GPUs contribute, all get full result (2P)
- **Reduce-scatter**: All contribute, each gets 1/N slice (P)
- **All-gather**: Each contributes slice, all get full result (P)
- **Broadcast**: One GPU sends to all (P)

### ZeRO Stage Implementations

#### Data Parallel (Baseline)
- Every GPU holds full model: weights, gradients, fp32 weights, optimizer state
- Gradients all-reduced every step
- Memory: 16 bytes/param × 30B = 480 GB total → 447 GB per GPU

#### ZeRO Stage 1
- Partitions optimizer state (fp32 weights + 2 running averages = 12 bytes)
- Each GPU keeps 1/N of optimizer state
- Still replicates weights and gradients
- Communication: Reduce-scatter + All-gather = 2P (same as DP)

#### ZeRO Stage 2
- Partitions optimizer state + gradients
- Each GPU keeps 1/N of gradients and optimizer state
- Still replicates fp32 weights
- Communication: Reduce-scatter grads + All-gather weights = 2P

#### ZeRO Stage 3
- Partitions everything: weights, gradients, fp32 weights, optimizer state
- Each GPU keeps only 1/N of each
- Forward pass: All-gather layer weights → compute → discard
- Backward pass: All-gather layer weights → compute → Reduce-scatter gradients
- Communication: 3P (forward all-gather + backward all-gather + reduce-scatter)

## Key Results (30B Model, 32 GPUs)

| Stage | Memory/GPU | Peak/GPU | Fits | Comm/Step | Comm Time (NVLink) |
|-------|------------|----------|------|-----------|-------------------|
| DP | 447 GB | 447 GB | ❌ | 120 GB | 267 ms |
| ZeRO-1 | 154 GB | 154 GB | ❌ | 120 GB | 267 ms |
| ZeRO-2 | 105 GB | 105 GB | ❌ | 120 GB | 267 ms |
| ZeRO-3 | 56 GB | 56 GB | ✅ | 180 GB | 400 ms |

At 32 GPUs:
- ZeRO-2: 105 GB/GPU → Still doesn't fit in 74.5 GB
- ZeRO-3: 56 GB/GPU → Fits with room for activations

## Understanding the Code

### Memory Calculation (from PDF)

Per parameter storage (16 bytes total):
```
Weight (fp16/bf16)        : 2 bytes
Gradient (fp16/bf16)      : 2 bytes
FP32 Master Weight        : 4 bytes
Optimizer State (2×fp32)  : 8 bytes
─────────────────────────────────
Total per parameter       : 16 bytes
```

For 30B parameters: 30B × 16 = 480 GB = 447 GiB

### Why ZeRO-1/2 Communication = 2P (Same as DP)

From the PDF Section 6:
> Stages 1 and 2 perform those same two phases and simply keep the intermediate slice instead of discarding it.

Data Parallel's all-reduce is internally implemented as reduce-scatter + all-gather. ZeRO-1/2 do the same two phases but keep the sharded result.

### Why ZeRO-3 Costs 3P

From the PDF Section 6:
> Stage 3 adds a further all-gather of the weights in the forward pass and again in the backward pass, which raises the total from 2P to 3P.

Forward: All-gather weights for each layer (P)
Backward: All-gather weights + Reduce-scatter gradients (2P)
Total: 3P

## V5 Decisions

Based on the arithmetic:

1. **Starting point**: ZeRO-2 from 32 GPUs or ZeRO-3 from 8 GPUs (DP and ZeRO-1 never fit)
2. **Communication logging**: Measure comm/compute ratio from step 1 (grows with faster hardware)
3. **Overlap enabled**: Gradient bucketing and communication overlap from start
4. **Open questions**:
   - ZeRO-2 on 32 vs ZeRO-3 on 8? (needs measured step time)
   - GPUs per node? (9x NVLink vs InfiniBand difference)
   - 8-bit from start? (needs BF16 vs MXFP8 comparison)
   - CPU offload? (depends on memory vs comm bound)

## Extending the Simulation

### Add New Model Sizes
```python
config = create_model_config(num_parameters=70_000_000_000, precision="fp16")
```

### Add FP8 Precision
```python
config = create_model_config(num_parameters=30_000_000_000, precision="fp8")
# Memory drops ~12% (from 447 GB to 393 GB for full model)
```

### Add Activation Recomputation
Modify `_estimate_activation_memory()` in simulators to account for checkpointing.

### Add CPU Offload
Extend `VirtualGPU` with CPU memory tier and simulate PCIe transfers.

## References

- **Session 12 PDF**: Distributed Training I, Data Parallel and ZeRO
- **ZeRO Paper**: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (Rajbhandari et al., 2019)
- **DeepSpeed**: Microsoft's ZeRO implementation
- **FSDP2**: PyTorch's Fully Sharded Data Parallel (ZeRO-3 equivalent)

## License

MIT License - Feel free to use for learning and research.