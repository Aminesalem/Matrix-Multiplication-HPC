#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// 1. i-j-k (Baseline)
void matmatijk(int ldA, int ldB, int ldC, double *A, double *B, double *C, int N, int M, int P) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j++) {
            for (int k = 0; k < M; k++) {
                C[i * ldC + j] += A[i * ldA + k] * B[k * ldB + j];
            }
        }
    }
}

// 2. i-k-j (The Cache-Friendly version)
void matmatikj(int ldA, int ldB, int ldC, double *A, double *B, double *C, int N, int M, int P) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < M; k++) {
            for (int j = 0; j < P; j++) {
                C[i * ldC + j] += A[i * ldA + k] * B[k * ldB + j];
            }
        }
    }
}

// 3. j-i-k
void matmatjik(int ldA, int ldB, int ldC, double *A, double *B, double *C, int N, int M, int P) {
    for (int j = 0; j < P; j++) {
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < M; k++) {
                C[i * ldC + j] += A[i * ldA + k] * B[k * ldB + j];
            }
        }
    }
}

// 4. j-k-i
void matmatjki(int ldA, int ldB, int ldC, double *A, double *B, double *C, int N, int M, int P) {
    for (int j = 0; j < P; j++) {
        for (int k = 0; k < M; k++) {
            for (int i = 0; i < N; i++) {
                C[i * ldC + j] += A[i * ldA + k] * B[k * ldB + j];
            }
        }
    }
}

// 5. k-i-j
void matmatkij(int ldA, int ldB, int ldC, double *A, double *B, double *C, int N, int M, int P) {
    for (int k = 0; k < M; k++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < P; j++) {
                C[i * ldC + j] += A[i * ldA + k] * B[k * ldB + j];
            }
        }
    }
}

// 6. k-j-i
void matmatkji(int ldA, int ldB, int ldC, double *A, double *B, double *C, int N, int M, int P) {
    for (int k = 0; k < M; k++) {
        for (int j = 0; j < P; j++) {
            for (int i = 0; i < N; i++) {
                C[i * ldC + j] += A[i * ldA + k] * B[k * ldB + j];
            }
        }
    }
}

// Timer function
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Helper function to clear matrix C before each test
void clear_matrix(double *C, int size) {
    memset(C, 0, sizeof(double) * size);
}

int main() {
    // Increased size slightly to make the performance differences more obvious
    int size = 512; 
    int N = size, M = size, P = size;
    int ldA = M, ldB = P, ldC = P;
    
    double *A = (double*)malloc(sizeof(double) * N * ldA);
    double *B = (double*)malloc(sizeof(double) * M * ldB);
    double *C = (double*)calloc(N * ldC, sizeof(double)); 

    for (int i = 0; i < N * ldA; i++) A[i] = 1.0;
    for (int i = 0; i < M * ldB; i++) B[i] = 1.0;

    double Nflops = 2.0 * N * M * P;
    double start, end, time_taken;

    printf("Benchmarking Matrix Multiplication (Size: %dx%d)\n", size, size);
    printf("--------------------------------------------------\n");

    // Test 1: ijk
    clear_matrix(C, N * ldC);
    start = get_time();
    matmatijk(ldA, ldB, ldC, A, B, C, N, M, P);
    end = get_time();
    printf("ijk version: \t%.4f Gflops\n", Nflops / (end - start) / 1e9);

    // Test 2: ikj
    clear_matrix(C, N * ldC);
    start = get_time();
    matmatikj(ldA, ldB, ldC, A, B, C, N, M, P);
    end = get_time();
    printf("ikj version: \t%.4f Gflops\n", Nflops / (end - start) / 1e9);

    // Test 3: jik
    clear_matrix(C, N * ldC);
    start = get_time();
    matmatjik(ldA, ldB, ldC, A, B, C, N, M, P);
    end = get_time();
    printf("jik version: \t%.4f Gflops\n", Nflops / (end - start) / 1e9);

    // Test 4: jki
    clear_matrix(C, N * ldC);
    start = get_time();
    matmatjki(ldA, ldB, ldC, A, B, C, N, M, P);
    end = get_time();
    printf("jki version: \t%.4f Gflops\n", Nflops / (end - start) / 1e9);

    // Test 5: kij
    clear_matrix(C, N * ldC);
    start = get_time();
    matmatkij(ldA, ldB, ldC, A, B, C, N, M, P);
    end = get_time();
    printf("kij version: \t%.4f Gflops\n", Nflops / (end - start) / 1e9);

    // Test 6: kji
    clear_matrix(C, N * ldC);
    start = get_time();
    matmatkji(ldA, ldB, ldC, A, B, C, N, M, P);
    end = get_time();
    printf("kji version: \t%.4f Gflops\n", Nflops / (end - start) / 1e9);

    free(A); free(B); free(C);
    return 0;
}