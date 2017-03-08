/*
 * Sum of a*X[N]
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include <omp.h>

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

#ifdef LOG_RESULTS
void log_result(char *algo, int numbers, int threads, double elapsed)
{
    FILE* f = fopen("results.txt", "a");
    fprintf(f, "%s,%d,%d,%f\n", algo, numbers, threads, elapsed * 1.0e3);
    fclose(f);
}
#else
#define log_result(algo, numbers, threads, elapsed)
#endif

#define REAL float
#define VECTOR_LENGTH 102400
int BASE = 128;
REAL total=0.0;
/* initialize a vector with random floating point numbers */
void init(REAL A[], int N) {
    int i;
    for (i = 0; i < N; i++) {
        A[i] = (double) drand48();
    }
}

int num_threads = 2;
REAL sum(int N, REAL X[], REAL a);
REAL sum_omp_parallel_task(int N, REAL X[], REAL a);
//REAL sum_omp_parallel_optimization(int N, REAL X[], REAL a);
int main(int argc, char *argv[]) {
    int N = VECTOR_LENGTH;
    if (argc < 2) {
        fprintf(stderr, "Usage: sum <n> (default %d) [<num_threads>] (default %d)\n", N, num_threads);
        exit(1);
    }
    N = atoi(argv[1]);
    if (argc >=3) num_threads = atoi(argv[2]);
    omp_set_num_threads(num_threads);
     REAL *X = malloc(sizeof(REAL)*N);

    srand48((1 << 12));
    init(X, N);
    REAL a = 0.1234;
    double tm_elaps;

#pragma omp parallel shared(N,X,a) num_threads(num_threads)
 {
   #pragma omp single 
    {
      int i; 
      int totalrun = 10; 
      double tm_begin = read_timer(); 
      for (i=0; i<totalrun; i++) 
        {total=0.0;
         total=sum_omp_parallel_task(N,X,a);
        }
      tm_elaps = read_timer() - tm_begin; 
      tm_elaps = tm_elaps/totalrun; 

      }
 }  
 log_result("parallel for", N, num_threads, parallel_task_elapsed);
 printf("======================================================================================================\n");
    printf("\tSum %d numbers using OpenMP with %d threads\n", N, num_threads);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS \n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Sum parallel task:\t%4f\t%4f\n", tm_elaps * 1.0e3, 2*N / (1.0e6 *tm_elaps)); 
//printf("Time for sum N=((%d) with %d threads): %f seconds\n", N, num_threads, (double)(tm_elaps)); 
 free(X);
 return 0;
}

REAL sum(int N, REAL X[], REAL a) {
    int i;
    REAL result = 0.0;
    for (i = 0; i < N; ++i)
        result += a * X[i];
    return result;
}

/*
 * Parallel reduction using OpenMP parallel for and reduction clause
 */
REAL sum_omp_parallel_task(int N, REAL X[], REAL a) {
REAL m,n;
    if (N<=BASE){ 
    int i; 
    REAL result=0.0; 
    for (i=0;i<N;++i) 
    result+=a *X[i];  
    return result; 
                } 
   else {
    #pragma omp task shared(m, N,X,a)
     m=sum_omp_parallel_task(N/2,X,a);
     n=sum_omp_parallel_task(N-N/2,X+N/2,a);
    #pragma omp taskwait
    return (n+m);
        }
}

