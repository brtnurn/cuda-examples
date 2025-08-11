#include <stdio.h>
#include <omp.h>

void kmer_cpu_parallel(char *ref, char *reads, int *counts, int ref_len, int read_len, int max_read_len, int read_count, int k)
{
    /*#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int thread_num = omp_get_num_threads();

            for (int i = 0; i < ref_len - k + 1; i++)
            {
                for (int j = tid; j < read_count; j += thread_num)
                {
                    for (int m = 0; m < read_len - k + 1; m++)
                    {
                        bool match = true;
                        for (int n = 0; n < k; n++)
                        {
                            if (ref[i + n] != reads[j * max_read_len + m + n])
                            {
                                match = false;
                                break;
                            }
                        }
                        if (match)
                        {
                            counts[j] += 1;
                        }
                    }
                }
            }
        }*/

#pragma omp parallel for schedule(static)
    for (int j = 0; j < read_count; ++j)
    {
        int c = 0;
        for (int i = 0; i < ref_len - k + 1; ++i)
            for (int m = 0; m < read_len - k + 1; ++m)
            {
                bool match = true;
                for (int n = 0; n < k; ++n)
                    if (ref[i + n] != reads[j * max_read_len + m + n])
                    {
                        match = false;
                        break;
                    }
                if (match)
                {
                    ++c;
                }
            }
        counts[j] = c;
    }
}