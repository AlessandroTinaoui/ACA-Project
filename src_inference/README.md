# KAN vs MLP on RISC-V with gem5

This repository compares C inference for spline-based KAN models and an MLP
baseline on a simulated RISC-V CPU using gem5.

The regression target is:

```text
f(x) = sin(2*pi*x) + 0.35*sin(10*pi*x)
```

All experiments use:

```text
N = 1024 points uniformly spaced in [0, 1]
```

Training is done outside this repository. This repo only handles:

- JSON model export loading through generated C headers;
- C inference;
- RISC-V compilation;
- gem5 simulation;
- metric comparison and plotting.

## Models

Models are stored in:

```text
model/
  1x1/
    mini_kan_riscv_export.json
  1x2x1/
    mini_kan_riscv_export.json
  1x4x1/
    mini_kan_riscv_export.json
  1x8x1/
    mini_kan_riscv_export.json
  mlp/
    mlp_export.json
```

Current models:

| Model | Architecture |
|---|---|
| KAN 1x1 | `[1, 1]` |
| KAN 1x2x1 | `[1, 2, 1]` |
| KAN 1x4x1 | `[1, 4, 1]` |
| KAN 1x8x1 | `[1, 8, 1]` |
| MLP | `[1, 24, 24, 1]` |

The KAN models use cubic B-splines:

```text
degree = 3
knots = 19
control points = 15
```

## Important Implementation Notes

Two important fixes/optimizations were made:

1. **Correctness fix:** spline inputs are no longer clamped to `[0, 1]` inside
   hidden layers. This was needed to match the pykan forward pass.
2. **Performance optimization:** the recursive Cox-de Boor B-spline evaluation
   was replaced by an iterative dynamic-programming version.

The KAN forward is implemented mainly in:

```text
src/kan_inference.c
src/bspline.c
```

`include/kan_model.h` and `include/mlp_model.h` are generated files. They are
created from JSON before compilation.

## Build and Run

### Run all models

This is the main command:

```bash
bash scripts/run_all_models_compare.sh 1024
```

It runs:

```text
KAN 1x1
KAN 1x2x1
KAN 1x4x1
KAN 1x8x1
MLP 24x24
```

and prints a comparison table.

### Compare existing results only

If the gem5 runs already exist:

```bash
python3 scripts/compare_all_models.py
```

### Plot results

Generate CSV and PNG plots:

```bash
python3 scripts/plot_model_metrics.py
```

Output:

```text
plots/model_metrics/metrics.csv
plots/model_metrics/*.png
```

The most useful plots are:

```text
mse.png
cycles.png
instructions.png
mse_vs_cycles.png
normalized_to_mlp.png
```

## Run One Model

Run one KAN model with L1+L2 cache:

```bash
bash scripts/run_cache.sh 1x4x1 1024
```

Other examples:

```bash
bash scripts/run_cache.sh 1x1 1024
bash scripts/run_cache.sh 1x2x1 1024
bash scripts/run_cache.sh 1x8x1 1024
```

Run the MLP:

```bash
bash scripts/run_mlp_l1_l2.sh model/mlp 1024
```

Compare one KAN against the MLP:

```bash
python3 scripts/compare_kan_mlp.py \
  --kan-dir results/cache/1x4x1 \
  --mlp-dir results/mlp_l1_l2
```

## Host-Only Sanity Checks

KAN:

```bash
python3 scripts/json_to_header.py 1x4x1
bash scripts/build_host.sh
./build/host/kan_demo_host 1024
```

MLP:

```bash
bash scripts/build_mlp_host.sh
./build/host/mlp_demo_host 1024
```

Debug fixed inputs:

```bash
bash scripts/build_debug_host.sh
./build/host/debug_compare_host
```

This prints:

```text
x,y_target,y_kan_c,y_mlp_c
```

plus selected KAN intermediate values.

## Results

Representative L1+L2 results after the correctness fix and iterative spline
optimization:

| Model | MSE | simInsts | cycles | IPC |
|---|---:|---:|---:|---:|
| KAN 1x1 | 2.35e-3 | 2.36M | 3.81M | 0.619 |
| KAN 1x2x1 | 1.26e-5 | 8.60M | 13.45M | 0.640 |
| KAN 1x4x1 | 7.99e-7 | 16.76M | 26.04M | 0.644 |
| KAN 1x8x1 | 4.16e-7 | 33.05M | 51.32M | 0.644 |
| MLP 24x24 | 4.87e-4 | 12.75M | 22.57M | 0.565 |

Main takeaway:

```text
KAN 1x2x1 is already much more accurate than the MLP and cheaper in cycles.
KAN 1x4x1 gives very high accuracy with moderate extra cost.
KAN 1x8x1 improves accuracy further but has diminishing returns.
```

## Useful Files

```text
KAN_MLP_REPORT.md          detailed technical report
OPTIMIZATION_ROADMAP.md    remaining optimization ideas
scripts/run_all_models_compare.sh
scripts/compare_all_models.py
scripts/plot_model_metrics.py
```

## gem5 Setup

The main comparison uses:

```text
RISC-V SE mode
Timing CPU
1 core
3 GHz
L1I 32KiB
L1D 32KiB
Private L2 256KiB
DDR3 512MiB
```

The scripts expect gem5 here:

```text
gem5/build/RISCV/gem5.opt
```
