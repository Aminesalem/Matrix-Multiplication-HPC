#include <stdio.h>
#include <stdlib.h>
#include <mpi.h> // The Message Passing Interface library

// Our standard fast loop, but modified so it only computes a specific "chunk" of the rows
void matmatikj_local(int N_local, int M, int P, double *A_local, double *B, double *C_local) {
    for (int i = 0; i < N_local; i++) {
        for (int k = 0; k < M; k++) {
            for (int j = 0; j < P; j++) {
                C_local[i * P + j] += A_local[i * M + k] * B[k * P + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    // 1. Wake up the network and figure out who is who
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);       // My specific ID (0 is the Master)
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);  // Total number of computers connected

    int size = 512;
    int N = size, M = size, P = size;
    
    // Calculate how many rows of matrix A each computer is responsible for
    int N_local = N / num_procs;

    double *A = NULL, *C = NULL;
    double *B = (double*)malloc(M * P * sizeof(double));
    
    // The smaller memory blocks that each individual computer needs
    double *A_local = (double*)malloc(N_local * M * sizeof(double));
    double *C_local = (double*)calloc(N_local * P, sizeof(double));

    // Only the Master (Rank 0) creates the full matrices
    if (rank == 0) {
        printf("MPI Matrix Multiplication (Size: %dx%d, Simulated Nodes: %d)\n", size, size, num_procs);
        printf("--------------------------------------------------\n");
        
        A = (double*)malloc(N * M * sizeof(double));
        C = (double*)calloc(N * P, sizeof(double));

        for (int i = 0; i < N * M; i++) A[i] = 1.0;
        for (int i = 0; i < M * P; i++) B[i] = 1.0;
    }

    // Start the synchronized timer
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // 2. Master broadcasts the FULL matrix B to every node on the network
    MPI_Bcast(B, M * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 3. Master chops up matrix A and SCATTERS the chunks to everyone
    MPI_Scatter(A, N_local * M, MPI_DOUBLE, A_local, N_local * M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 4. Every node (including the master) computes its own piece simultaneously
    matmatikj_local(N_local, M, P, A_local, B, C_local);

    // 5. Master GATHERS all the finished chunks of C back together
    MPI_Gather(C_local, N_local * P, MPI_DOUBLE, C, N_local * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    // Only the Master prints the final results
    if (rank == 0) {
        double time_taken = end_time - start_time;
        double nflops = 2.0 * N * M * P;
        printf("Time taken: \t%.4f seconds\n", time_taken);
        printf("Performance: \t%.4f Gflops\n", nflops / time_taken / 1e9);
        
        free(A); free(C);
    }

    // Everyone cleans up their local memory
    free(B); free(A_local); free(C_local);
    
    // Shut down the network
    MPI_Finalize();
    return 0;
}