#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_SIZE 16

__global__ void matmat_tiled(int N, double *A, double *B, double *C) {
    __shared__ double tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ double tile_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    double sum = 0.0;

    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        if (row < N && m * TILE_SIZE + tx < N) tile_A[ty][tx] = A[row * N + m * TILE_SIZE + tx];
        else tile_A[ty][tx] = 0.0;

        if (m * TILE_SIZE + ty < N && col < N) tile_B[ty][tx] = B[(m * TILE_SIZE + ty) * N + col];
        else tile_B[ty][tx] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) sum += tile_A[ty][k] * tile_B[k][tx];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = sum;
}

void matmat_cpu(int N, double *A, double *B, double *C) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

int main() {
    printf("Size\tCPU_Time(s)\tTiled_GPU_Time(s)\tValidation\n");
    printf("----------------------------------------------------------\n");

    for (int N = 256; N <= 768; N += 256) {
        size_t bytes = N * N * sizeof(double);
        double *h_A = (double*)malloc(bytes);
        double *h_B = (double*)malloc(bytes);
        double *h_C_cpu = (double*)calloc(N * N, sizeof(double));
        double *h_C_gpu = (double*)calloc(N * N, sizeof(double));

        for (int i = 0; i < N * N; i++) { h_A[i] = 1.0; h_B[i] = 1.0; }

        double *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

        cudaEvent_t start, stop; float ms;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        cudaEventRecord(start);
        matmat_cpu(N, h_A, h_B, h_C_cpu);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        double cpu_time = ms / 1000.0;

        cudaMemset(d_C, 0, bytes);
        cudaEventRecord(start);
        matmat_tiled<<<blocks, threads>>>(N, d_A, d_B, d_C);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        double gpu_time = ms / 1000.0;

        cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);

        bool is_correct = true;
        for (int i = 0; i < N * N; i++) {
            if (fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-5) is_correct = false;
        }

        printf("%d\t%.4f\t\t%.4f\t\t\t%s\n", N, cpu_time, gpu_time, is_correct ? "PASSED" : "FAILED");

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    }
    return 0;
}