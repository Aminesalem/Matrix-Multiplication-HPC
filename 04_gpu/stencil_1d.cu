#include <stdio.h>
#include <stdlib.h>

#define RADIUS 1
#define BLOCK_SIZE 256

__global__ void stencil_1d(int *in, int *out, int N) {
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;

    if (gindex < N) temp[lindex] = in[gindex];
    if (threadIdx.x < RADIUS) {
        temp[lindex - RADIUS] = (gindex >= RADIUS) ? in[gindex - RADIUS] : 0;
        temp[lindex + BLOCK_SIZE] = (gindex + BLOCK_SIZE < N) ? in[gindex + BLOCK_SIZE] : 0;
    }
    __syncthreads();

    if (gindex < N) out[gindex] = temp[lindex - 1] + temp[lindex] + temp[lindex + 1];
}

void stencil_cpu(int *in, int *out, int N) {
    for (int i = 0; i < N; i++) {
        int left = (i >= RADIUS) ? in[i - RADIUS] : 0;
        int right = (i + RADIUS < N) ? in[i + RADIUS] : 0;
        out[i] = left + in[i] + right;
    }
}

int main() {
    int N = 10000;
    size_t bytes = N * sizeof(int);
    int *h_in = (int*)malloc(bytes), *h_out_cpu = (int*)malloc(bytes), *h_out_gpu = (int*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = 1;

    int *d_in, *d_out;
    cudaMalloc(&d_in, bytes); cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    stencil_1d<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_in, d_out, N);
    cudaMemcpy(h_out_gpu, d_out, bytes, cudaMemcpyDeviceToHost);

    stencil_cpu(h_in, h_out_cpu, N);

    int errors = 0;
    for (int i = 0; i < N; i++) if (h_out_cpu[i] != h_out_gpu[i]) errors++;

    if (errors == 0) printf("Advanced Shared Memory Stencil: PASSED VALIDATION\n");
    else printf("Validation Failed: %d errors\n", errors);

    cudaFree(d_in); cudaFree(d_out); free(h_in); free(h_out_cpu); free(h_out_gpu);
    return 0;
}