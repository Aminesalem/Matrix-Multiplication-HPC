#include <stdio.h>
#include <stdlib.h>

#define WIDTH 64
#define HEIGHT 64

__global__ void pitched_kernel(float* devPtr, size_t pitch, int width, int height) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < height && c < width) {
        float* row = (float*)((char*)devPtr + r * pitch);
        row[c] = row[c] * 2.0f;
    }
}

int main() {
    size_t size = WIDTH * HEIGHT * sizeof(float);
    float *h_data = (float*)malloc(size), *h_result = (float*)malloc(size);
    for (int i = 0; i < WIDTH * HEIGHT; i++) h_data[i] = 1.0f;

    float *d_data; size_t pitch;
    cudaMallocPitch(&d_data, &pitch, WIDTH * sizeof(float), HEIGHT);
    cudaMemcpy2D(d_data, pitch, h_data, WIDTH * sizeof(float), WIDTH * sizeof(float), HEIGHT, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((WIDTH + threads.x - 1) / threads.x, (HEIGHT + threads.y - 1) / threads.y);
    pitched_kernel<<<blocks, threads>>>(d_data, pitch, WIDTH, HEIGHT);
    cudaMemcpy2D(h_result, WIDTH * sizeof(float), d_data, pitch, WIDTH * sizeof(float), HEIGHT, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < WIDTH * HEIGHT; i++) if (h_result[i] != 2.0f) errors++;

    if (errors == 0) printf("Pitched Memory / cudaMemcpy2D: PASSED VALIDATION\n");
    else printf("Validation Failed!\n");

    cudaFree(d_data); free(h_data); free(h_result);
    return 0;
}