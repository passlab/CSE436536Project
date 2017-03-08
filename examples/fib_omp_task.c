#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <math.h>
double read_timer_ms() {
    struct timeval t;
    gettimeofday(&t, 0);
    return t.tv_sec * 1000ULL + t.tv_usec / 1000ULL;
}

double read_timer() {
    return read_timer_ms() * 1.0e-3;
}

int fib(int n)
{
  int x, y;
  if (n<2)return n;
  else {
       #pragma omp task shared(x) firstprivate(n)
       x=fib(n-1);

       #pragma omp task shared(y) firstprivate(n)
       y=fib(n-2);

       #pragma omp taskwait
       return (x+y);
    }
}

int main(int argc, char * argv[])
{
  int n, result;

  if (argc < 2) {
        fprintf(stderr, "Usage: fib <n> [<#threads>]\n");
        exit(1);
  }
  n = atoi(argv[1]);
  int num_threads;
  if (argc > 2) {
	num_threads  = atoi(argv[2]);
	omp_set_num_threads(num_threads);
  }

  n = atoi(argv[1]);
 #pragma omp parallel shared(n, num_threads, result)
  {
    #pragma omp single
    {
    	num_threads = omp_get_num_threads();
	result = fib(n);
    }
  }
  printf ("fib(%d) = %d\n", n, fib(n));
  return 0;
}
