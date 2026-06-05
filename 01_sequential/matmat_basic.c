#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Version 1: The standard i-j-k matrix multiplication
// This uses the prototype exactly as requested by the assignment.
void matmatijk(int ldA, int ldB, int ldC, double *A, double *B, double *C, int N, int M, int P) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j++) {
            for (int k = 0; k < M; k++) {
                // Accessing 1D array as a 2D matrix using leading dimensions
                C[i * ldC + j] += A[i * ldA + k] * B[k * ldB + j];
            }
        }
    }
}

// A simple helper function to get the current time in seconds
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    // Let's test with a small matrix size first (256x256)
    int size = 256;
    int N = size, M = size, P = size;
    
    // Leading dimensions (usually just the width of the matrix)
    int ldA = M, ldB = P, ldC = P;
    
    // Allocate memory for our matrices
    double *A = (double*)malloc(sizeof(double) * N * ldA);
    double *B = (double*)malloc(sizeof(double) * M * ldB);
    // calloc sets everything to 0 initially, which is important for our result matrix C
    double *C = (double*)calloc(N * ldC, sizeof(double)); 

    // Fill matrices A and B with some dummy data (just 1.0 everywhere)
    for (int i = 0; i < N * ldA; i++) A[i] = 1.0;
    for (int i = 0; i < M * ldB; i++) B[i] = 1.0;

    // Time the multiplication
    printf("Starting multiplication for size %d...\n", size);
    
    double start_time = get_time();
    matmatijk(ldA, ldB, ldC, A, B, C, N, M, P);
    double end_time = get_time();
    
    double time_taken = end_time - start_time;

    // Calculate Gigaflops (Gflops)
    double Nflops = 2.0 * N * M * P;
    double gflops = Nflops / time_taken / 1e9;

    printf("Time taken: %f seconds\n", time_taken);
    printf("Performance: %f Gflops\n", gflops);

    // Clean up memory
    free(A);
    free(B);
    free(C);

    return 0;
}