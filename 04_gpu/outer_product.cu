#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define BLOCK_SIZE 16

__global__ void outer_product(float *A, float *B, float *C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < size && col < size) C[row * size + col] = A[row] * B[col];
}

void cpu_outer_product(float *A, float *B, float *C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) C[i * size + j] = A[i] * B[j];
    }
}

int main() {
    size_t vec_bytes = N * sizeof(float);
    size_t mat_bytes = N * N * sizeof(float);
    float *h_A = (float*)malloc(vec_bytes), *h_B = (float*)malloc(vec_bytes);
    float *h_C_gpu = (float*)malloc(mat_bytes), *h_C_cpu = (float*)malloc(mat_bytes);

    for(int i=0; i<N; i++) { h_A[i] = i % 10; h_B[i] = i % 5; }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, vec_bytes); cudaMalloc(&d_B, vec_bytes); cudaMalloc(&d_C, mat_bytes);
    cudaMemcpy(d_A, h_A, vec_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, vec_bytes, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    outer_product<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C_gpu, d_C, mat_bytes, cudaMemcpyDeviceToHost);

    cpu_outer_product(h_A, h_B, h_C_cpu, N);

    int errors = 0;
    for(int i=0; i<N*N; i++) if(h_C_cpu[i] != h_C_gpu[i]) errors++;

    if(errors == 0) printf("Outer Product 2D Kernel: PASSED VALIDATION\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
    return 0;
}