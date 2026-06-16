#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void matmat_kernel(int N, int M, int P, double *A, double *B, double *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < P) {
        double sum = 0.0;
        for (int k = 0; k < M; k++) {
            sum += A[row * M + k] * B[k * P + col];
        }
        C[row * P + col] = sum;
    }
}

int main() {
    int size = 1024; 
    int N = size, M = size, P = size;
    size_t bytes_A = N * M * sizeof(double);
    size_t bytes_B = M * P * sizeof(double);
    size_t bytes_C = N * P * sizeof(double);

    double *h_A = (double*)malloc(bytes_A);
    double *h_B = (double*)malloc(bytes_B);
    double *h_C = (double*)malloc(bytes_C);

    for (int i = 0; i < N * M; i++) h_A[i] = 1.0;
    for (int i = 0; i < M * P; i++) h_B[i] = 1.0;

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((P + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    printf("Launching CUDA Kernel on Cloud NVIDIA T4 GPU (Size: %dx%d)...\n", size, size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matmat_kernel<<<numBlocks, threadsPerBlock>>>(N, M, P, d_A, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);

    double seconds = milliseconds / 1000.0;
    double nflops = 2.0 * N * M * P;
    double gflops = nflops / seconds / 1e9;

    printf("Time taken: \t%.4f seconds\n", seconds);
    printf("Performance: \t%.4f Gflops\n", gflops);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}