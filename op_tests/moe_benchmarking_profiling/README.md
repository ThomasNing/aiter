# MOE Benchmarking & Profiling Pipeline

Production-quality scripts for MOE kernel benchmarking, profiling, and performance analysis.

## Scripts

| Script | Purpose |
|--------|---------|
| `extract_configs.py` | Extract config from PyTorch trace → configs/ |
| `benchmark_and_analyze.py` | Benchmark + select best kernels → results/ |
| `profile_kernels.py` | Profile with rocprofv3 → results/profiling/ |
| `analyze_profiling.py` | Generate performance analyses → results/analysis/ |
| `analysis_notebook.ipynb` | Interactive Jupyter notebook for visualization |
| `moe_utils.py` | Shared utilities |

## Quick Start

```bash
cd aiter/op_tests/moe_benchmarking_profiling

# Run with defaults (uses configs/ and saves to results/)
python benchmark_and_analyze.py
python profile_kernels.py -i results/benchmark_best_kernels.csv
python analyze_profiling.py -i results/profiling/kernels_with_counters.csv

# Or use the interactive notebook
jupyter notebook analysis_notebook.ipynb
```

## Outputs

- `results/benchmark_*.csv` - Benchmark results, combinations, best kernels
- `results/profiling/kernels_with_counters.csv` - Profiling data
- `results/analysis/*.html` - Roofline plots

## Useful Options

```bash
# Custom config file
python benchmark_and_analyze.py -i configs/example_config.csv

# Stricter error filtering
python benchmark_and_analyze.py --error-threshold 30.0

# Use MI350 specs
python analyze_profiling.py --gpu MI350 -i results/profiling/kernels_with_counters.csv

# Keep temp files for debugging
python profile_kernels.py --keep-temp-files -i results/benchmark_best_kernels.csv
```

## Dependencies

```bash
pip install torch pandas numpy plotly einops
```

Profiling requires `rocprofv3` on system.
