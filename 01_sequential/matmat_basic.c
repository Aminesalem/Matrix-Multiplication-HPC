#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// The winning cache-friendly loop
void matmatikj(int ldA, int ldB, int ldC, double *A, double *B, double *C, int N, int M, int P) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < M; k++) {
            for (int j = 0; j < P; j++) {
                C[i * ldC + j] += A[i * ldA + k] * B[k * ldB + j];
            }
        }
    }
}

// The blocked matrix multiplication
void matmatblock(int ldA, int ldB, int ldC, double *A, double *B, double *C, 
                 int N, int M, int P, int dbA, int dbB, int dbC) {
    
    // Iterate over the matrix in blocks
    for (int i = 0; i < N; i += dbA) {
        for (int k = 0; k < M; k += dbB) {
            for (int j = 0; j < P; j += dbC) {
                
                // Handle edge cases where the block doesn't perfectly divide the matrix size
                int current_N = (i + dbA > N) ? (N - i) : dbA;
                int current_M = (k + dbB > M) ? (M - k) : dbB;
                int current_P = (j + dbC > P) ? (P - j) : dbC;

                // Pass the correct pointer offsets using the leading dimensions
                matmatikj(ldA, ldB, ldC, 
                          A + i * ldA + k, 
                          B + k * ldB + j, 
                          C + i * ldC + j, 
                          current_N, current_M, current_P);
            }
        }
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void clear_matrix(double *C, int size) {
    memset(C, 0, sizeof(double) * size);
}

int main() {
    int start_size = 256;
    int end_size = 1536;
    int step = 256;
    
    // Set block sizes (dbA = dbB = dbC)
    int dbA = 64, dbB = 64, dbC = 64; 

    printf("N\tGflops (ikj)\tGflops (block)\n");
    printf("--------------------------------------\n");

    for (int size = start_size; size <= end_size; size += step) {
        int N = size, M = size, P = size;
        int ldA = M, ldB = P, ldC = P;
        
        double *A = (double*)malloc(sizeof(double) * N * ldA);
        double *B = (double*)malloc(sizeof(double) * M * ldB);
        double *C = (double*)calloc(N * ldC, sizeof(double)); 

        for (int i = 0; i < N * ldA; i++) A[i] = 1.0;
        for (int i = 0; i < M * ldB; i++) B[i] = 1.0;

        double Nflops = 2.0 * N * M * P;
        double start, end, time_ikj, time_block;

        // Test matmatikj
        clear_matrix(C, N * ldC);
        start = get_time();
        matmatikj(ldA, ldB, ldC, A, B, C, N, M, P);
        end = get_time();
        time_ikj = end - start;

        // Test matmatblock
        clear_matrix(C, N * ldC);
        start = get_time();
        matmatblock(ldA, ldB, ldC, A, B, C, N, M, P, dbA, dbB, dbC);
        end = get_time();
        time_block = end - start;

        printf("%d\t%.4f\t\t%.4f\n", size, Nflops / time_ikj / 1e9, Nflops / time_block / 1e9);

        free(A); free(B); free(C);
    }

    return 0;
}