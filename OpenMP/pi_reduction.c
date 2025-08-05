#include <stdio.h>
#include <omp.h>

#define STEPS 10000
#define THREADS 10
int main()
{
    double result, sum = 0;
    double step = 1.0 / STEPS;
    omp_set_num_threads(THREADS);
    omp_set_schedule(omp_sched_static, 1);
    double start = omp_get_wtime();

#pragma omp parallel
    {
        double x;
#pragma omp for reduction(+ : sum) schedule(runtime)
        for (int i = 0; i < STEPS; i++)
        {
            x = (i + 0.5) * step;
            sum += 4 / (1 + x * x);
        }
    }
    double stop = omp_get_wtime();
    double time = (stop - start) * 1000;
    result = sum * step;
    printf("Result: %f\n", result);
    printf("Time: %fms\n", time);
    return 0;
}