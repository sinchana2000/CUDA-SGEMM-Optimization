# CUDA SGEMM Optimization

Hand-written CUDA matrix-multiply kernels, taken from a naive version up toward
library speed on an NVIDIA T4 and benchmarked against cuBLAS.

## Results (Tesla T4 on Google Colab, single precision)

M = N = K = 2048:

| Kernel | GFLOP/s | % of cuBLAS |
|---|---|---|
| naive | 435 | 9.0% |
| tiled (shared memory) | 899 | 18.7% |
| reg1d (1D register tiling) | 2,075 | 43.1% |
| cuBLAS | 4,814 | 100% |

All sizes from the same run (every kernel matches the cuBLAS output with relative error below 2e-6):

| Size | naive | tiled | reg1d | cuBLAS |
|---|---|---|---|---|
| 512 | 377 | 617 | 1,220 | 3,304 |
| 1024 | 400 | 925 | 2,094 | 4,205 |
| 2048 | 435 | 899 | 2,075 | 4,814 |

Numbers are GFLOP/s from one Colab session (3 warmup + 20 timed iterations per kernel). Run-to-run variation on the shared T4 is roughly 5%: a second run gave 1,960 GFLOP/s (40.1% of cuBLAS) for reg1d at 2048.

### Fusing the epilogue

`ReLU(A*B + bias)` at 2048: a GEMM followed by a separate bias+ReLU pass takes 19.5 ms, the fused kernel takes 19.0 ms (1.03x). The gain is small because the GEMM itself dominates the runtime, and the fused kernel is built on the shared-memory tiled kernel, so it is bounded by that kernel's speed.

## Run it

Open `cuda_gemm_colab.ipynb` in Google Colab, set the runtime to a T4 GPU, and run
all cells. Or, on any machine with an NVIDIA GPU and the CUDA toolkit:

```bash
nvcc -O3 -arch=sm_75 gemm.cu -o gemm -lcublas
./gemm
```

Change `-arch` to match your GPU (`sm_75` is Turing, the T4).

## What each kernel does

- **naive** - one thread per output element, no reuse, memory bound
- **tiled** - stages 32x32 tiles of A and B in shared memory, reused across the block
- **reg1d** - 64x64 output tile per block, each thread computes 8 results in registers for more reuse
- **fused GEMM + bias + ReLU** - folds the activation into the GEMM epilogue to skip an extra pass over the output

## Next steps

- 2D register tiling (each thread computes a small square tile) to cut shared-memory traffic further
- Vectorized `float4` loads and double buffering
- Tensor cores (WMMA) with FP16 inputs on the T4

## Files

- `gemm.cu` - all kernels and the benchmark harness
- `cuda_gemm_colab.ipynb` - one-click Colab runner

## Built on

An NVIDIA T4 (Turing, sm_75), CUDA, cuBLAS, and Nsight Compute for profiling.
