# CUDA SGEMM Optimization

Hand-written CUDA matrix-multiply kernels, taken from a naive version up toward
library speed on an NVIDIA T4 and benchmarked against cuBLAS.

## Results (Tesla T4, 2048 x 2048 x 2048)

| Kernel | GFLOP/s | % of cuBLAS |
|---|---|---|
| naive | 405 | 9% |
| tiled (shared memory) | 847 | 19% |
| reg1d (1D register tiling) | 1,837 | 42% |
| cuBLAS | 4,421 | 100% |

## Run it

Open `cuda_gemm_colab.ipynb` in Google Colab, set the runtime to a T4 GPU, and run
all cells. Or, on any machine with an NVIDIA GPU and the CUDA toolkit:

```bash
nvcc -O3 -arch=sm_75 gemm.cu -o gemm -lcublas
./gemm
```

## What each kernel does

- **naive** - one thread per output element, no reuse, memory bound
- **tiled** - stages tiles of A and B in shared memory, reused across the block
- **reg1d** - each thread computes 8 results in registers for more reuse
- **reg2d** - each thread computes an 8x8 tile, attacking the shared-memory bottleneck
- **fused GEMM + bias + ReLU** - folds the activation into the GEMM epilogue to skip an extra pass over the output

## Files

- `gemm.cu` - all kernels and the benchmark harness
- `cuda_gemm_colab.ipynb` - one-click Colab runner
- `NOTES.md` - the ideas behind each optimization

## Built on

An NVIDIA T4 (Turing, sm_75), CUDA, cuBLAS, and Nsight Compute for profiling.
