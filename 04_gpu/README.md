# HPC Matrix Multiplication Assignment

This repository contains my code for the High Performance Computing matrix multiplication assignment. The goal of this project was to implement and scale matrix multiplication starting from a basic single-core CPU version all the way up to a GPU-accelerated version.

## Project Structure

I broke the assignment down into four main parts:

### 1. Sequential & Cache Optimization (`01_sequential`)
- Tested all 6 loop permutations (`ijk`, `ikj`, `jik`, etc.) to see how CPU caching affects performance.
- The `ikj` and `kij` versions were the fastest because they access memory linearly.
- Implemented blocked matrix multiplication (`matmatblock`) to handle larger matrix sizes efficiently without dropping out of the CPU cache.
- **Compile:** `gcc -O3 matmat_basic.c -o matmat_basic`

### 2. Shared Memory / Multithreading (`02_multithreaded`)
- Used OpenMP to parallelize the outermost loop of the fastest sequential version (`ikj`).
- This distributed the workload across all 8 threads on my local CPU, providing a massive speedup over the single-core baseline.
- **Compile:** `gcc -O3 -fopenmp matmat_omp.c -o matmat_omp`

### 3. Distributed Memory (`03_distributed`)
- Used MPI to simulate a distributed cluster environment.
- The master node scatters chunks of Matrix A to the worker nodes, broadcasts Matrix B, and then gathers the computed chunks of Matrix C back together.
- **Compile:** `mpicc -O3 matmat_mpi.c -o matmat_mpi`
- **Run:** `mpirun -np 4 ./matmat_mpi`

### 4. GPU Acceleration (`04_gpu`)
- Wrote a CUDA kernel to offload the multiplication to the GPU using 16x16 thread blocks.
- **Hardware Note:** My local machine only has an Intel integrated GPU, which cannot run CUDA code natively. To compile and test this section, I used a Google Colab notebook with an NVIDIA T4 GPU instance. The code in `matmat_gpu.cu` was tested and verified there, achieving ~17.5 Gflops.
- **Compile:** `nvcc -O3 matmat_gpu.cu -o matmat_gpu`

## Performance Summary
Starting from a poorly optimized `jki` loop running at ~0.3 Gflops, the code scaled up through OpenMP and MPI, finally peaking at ~17.5 Gflops using the CUDA implementation on the cloud GPU.
