#include <stdio.h>
#include <omp.h>

#define STEPS 10000
#define THREADS 10
int main()
{
    double result = 0;
    double step = 1.0 / STEPS;
    double sums[THREADS];
    omp_set_num_threads(THREADS);
    double start = omp_get_wtime();
#pragma omp parallel
    {
        int id = omp_get_thread_num();
        sums[id] = 0.0;
        int thread_num = omp_get_num_threads();
        double x, sum = 0;

        for (int i = id; i < STEPS; i += thread_num)
        {
            x = (i + 0.5) * step;
            sum += 4 / (1 + x * x);
        }

        sums[id] = sum * step;
    }
    double stop = omp_get_wtime();
    double time = stop - start;
    for (int i = 0; i < THREADS; i++)
    {
        result += sums[i];
    }
    printf("Result: %f\n", result);
    printf("Time: %f\n", time);
    return 0;
}