/*
 * Square matrix multiplication
 * A[N][N] * B[N][N] = C[N][N]
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>
#include <string.h>
#include "omp.h"

int BASE;

/* read timer in second */
double read_timer() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time + (double) tm.millitm / 1000.0;
}

/* read timer in ms */
double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

#define REAL float

void init(int N, REAL *A) {
    int i, j;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i*N+j] = (REAL) drand48();
        }
    }
}

           
void matmul_omp_task(int N, int tempN, REAL *A, REAL *B, REAL *C)  {

    //int i, j, k;
     //REAL temp;
    if (N<=BASE) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < tempN; j++) {
                 REAL temp = 0;
                for (int k = 0; k < tempN; k++) {
                    temp += (A[i * tempN + k] * B[k * tempN + j]);
                }
                C[i * tempN + j] = temp;
            }
        }
    } else {
#pragma omp task shared(N,tempN,A,B,C)
        matmul_omp_task(N/2, tempN, A,B,C);
        matmul_omp_task(N-N/2, tempN, A+N/2, B+N/2, C+N/2);
#pragma omp taskwait
    }

}


int main(int argc, char *argv[]) {
    int N;

    int num_threads = 8; /* 4 is default number of threads */
    if (argc < 2) {
        fprintf(stderr, "Usage: axpy <n> (default %d) [<num_threads>] (default %d)\n", N, num_threads);
        exit(1);
    }
    N = atoi(argv[1]);
    if (argc >=3) num_threads = atoi(argv[2]);
    omp_set_num_threads(num_threads);
    BASE = N/num_threads;

    double elapsed_omp;

    REAL *A = (REAL *)malloc(sizeof(REAL)*N*N);
    REAL *B = (REAL *)malloc(sizeof(REAL)*N*N);
    REAL *C_omp = (REAL *)malloc(sizeof(REAL)*N*N);

    srand48((1 << 12));
    init(N, A);
    init(N, B);

    int i;
    int num_runs = 10;

    elapsed_omp = read_timer();
    for (i=0; i<num_runs; i++) {
#pragma omp parallel shared(N,A,B,C_omp) num_threads(num_threads)
        {
#pragma omp single
            matmul_omp_task(N, N, A, B, C_omp);
#pragma omp taskwait
        }
    }
    elapsed_omp = (read_timer() - elapsed_omp)/num_runs;
    /* you should add the call to each function and time the execution */

    printf("======================================================================================================\n");
    printf("\tMatrix Multiplication: A[N][N] * B[N][N] = C[N][N], N=%d\n", N);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS\n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("matmul_omp_task:\t\t%4f\t%4f\n", elapsed_omp * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_omp)));
    return 0;
}



