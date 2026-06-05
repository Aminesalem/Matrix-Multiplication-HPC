    #include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h> // The OpenMP library

// Our winning cache-friendly loop, now supercharged with multithreading
void matmat_omp_ikj(int ldA, int ldB, int ldC, double *A, double *B, double *C, int N, int M, int P) {
    
    // This directive tells the compiler to divide the 'i' iterations across all available CPU cores
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < M; k++) {
            for (int j = 0; j < P; j++) {
                C[i * ldC + j] += A[i * ldA + k] * B[k * ldB + j];
            }
        }
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    // We need a larger matrix (1024x1024) to truly see the benefit of multiple cores
    int size = 1024; 
    int N = size, M = size, P = size;
    int ldA = M, ldB = P, ldC = P;
    
    double *A = (double*)malloc(sizeof(double) * N * ldA);
    double *B = (double*)malloc(sizeof(double) * M * ldB);
    double *C = (double*)calloc(N * ldC, sizeof(double)); 

    for (int i = 0; i < N * ldA; i++) A[i] = 1.0;
    for (int i = 0; i < M * ldB; i++) B[i] = 1.0;

    double Nflops = 2.0 * N * M * P;

    printf("Benchmarking OpenMP Multithreaded Matrix Multiplication (Size: %dx%d)\n", size, size);
    
    // Let's ask OpenMP how many cores it found on your specific machine
    int max_threads = omp_get_max_threads();
    printf("Max available CPU threads: %d\n", max_threads);
    printf("--------------------------------------------------\n");

    double start = get_time();
    matmat_omp_ikj(ldA, ldB, ldC, A, B, C, N, M, P);
    double end = get_time();
    
    double time_taken = end - start;
    double gflops = Nflops / time_taken / 1e9;

    printf("Time taken: \t%.4f seconds\n", time_taken);
    printf("Performance: \t%.4f Gflops\n", gflops);

    free(A); free(B); free(C);
    return 0;
}