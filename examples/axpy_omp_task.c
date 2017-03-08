/*
 * AXPY  Y[N] = Y[N] + a*X[N]
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>

#include <omp.h>
int BASE = 64 ;
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
#define VECTOR_LENGTH 102400

/* initialize a vector with random floating point numbers */
void init(REAL *A, int N) {
    int i;
//#pragma omp parallel for shared(A, N) private(i)
    for (i = 0; i < N; i++) {
        A[i] = (double) drand48();
    }
}

double check(REAL *A, REAL B[], int N) {
    int i;
    double sum = 0.0;
    for (i = 0; i < N; i++) {
        sum += A[i] - B[i];
    }
    return sum;
}


void axpy_omp_task_driver(int N, REAL *Y, REAL *X, REAL a, int num_threads);
void axpy_omp_task(int N, REAL *Y, REAL *X, REAL a);
int main(int argc, char *argv[]) {
    int N = VECTOR_LENGTH;
    int num_threads = 4; /* 4 is default number of threads */
    if (argc < 2) {
        fprintf(stderr, "Usage: axpy <n> [<#threads(%d)>] (n should be dividable by #threads)\n", num_threads);
        exit(1);
    }
    N = atoi(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);
    omp_set_num_threads(num_threads);
    REAL a = 123.456;
    REAL *Y_base = malloc(sizeof(REAL)*N);
    REAL *Y_parallel = malloc(sizeof(REAL)*N);
    REAL *X = malloc(sizeof(REAL)* N);

    srand48((1 << 12));
    init(X, N);
    init(Y_base, N);
    memcpy(Y_parallel, Y_base, N * sizeof(REAL));
    /* example run */

    int i;
    int num_runs = 10;
        
    double elapsed_omp_task = read_timer();
    for (i=0; i<num_runs; i++) axpy_omp_task_driver(N, Y_parallel, X, a, num_threads);
    elapsed_omp_task = (read_timer() - elapsed_omp_task)/num_runs;
    
    
    /* you should add the call to each function and time the execution */
    printf("======================================================================================================\n");
    printf("\tAXPY: Y[N] = Y[N] + a*X[N], N=%d, %d threads for dist\n", N, num_threads);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\t\tRuntime (ms)\t MFLOPS \t\tError (compared to base)\n");
    printf("------------------------------------------------------------------------------------------------------\n");
        printf("axpy_omp_task:\t\t\t%4f\t%4f \t\t%g\n", elapsed_omp_task * 1.0e3, (2.0 * N) / (1.0e6 * elapsed_omp_task), check(Y_base,
                                                                                                              Y_parallel, N));

    free(Y_base);
    free(Y_parallel);
    free(X);

    return 0;
}

void axpy_omp_task_driver(int N, REAL *Y, REAL *X, REAL a, int num_threads) {
#pragma omp parallel shared(N, X, Y, a)  num_threads(num_threads)
{
#pragma omp single
axpy_omp_task(N, Y, X, a);
}
}
//=============continuation scheduler
void axpy_omp_task(int N, REAL *Y, REAL *X, REAL a) {
  if (N<=BASE) {
    int i;
    for (i = 0; i < N; ++i)
      Y[i] += a * X[i];
  } else {
    #pragma omp task shared(Y, X, a, N)
    axpy_omp_task(N/2, Y, X, a);
    axpy_omp_task(N-N/2, Y+N/2, X+N/2, a);
    #pragma omp taskwait
  }
}

void axpy_omp_parallel_for(int N, REAL *Y, REAL *X, REAL a) {
    int i;
    #pragma omp parallel shared(N, X, Y, a) private(i)
    #pragma omp for
    for (i = 0; i < N; ++i)
        Y[i] += a * X[i];
}
