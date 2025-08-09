#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <stdbool.h>

#define MAX_REF_LEN 1000000
#define MAX_READ_LEN 200
#define MAX_READ 20480
#define THREADS 8

void kmer_cpu_parallel(char *ref, char *reads, int *counts, int ref_len, int read_len, int max_read_len, int read_count, int k)
{
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int thread_num = omp_get_num_threads();
        int kmer_count = read_len - k + 1;

        for (int i = 0; i < ref_len - k + 1; i++)
        {
            for (int j = tid; j < read_count; j += thread_num)
            {
                for (int m = 0; m < kmer_count; m++)
                {
                    bool flag = true;
                    for (int n = 0; n < k; n++)
                    {
                        if (ref[i + n] != reads[j * max_read_len + m + n])
                        {
                            flag = false;
                            break;
                        }
                    }
                    if (flag)
                    {
                        counts[j] += 1;
                    }
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        printf("Incorrect usage");
        return -1;
    }

    char *ref_file = argv[1];
    char *reads_file = argv[2];
    int k = atoi(argv[3]);
    char *out_file = argv[4];

    FILE *fptr;

    char ref[MAX_REF_LEN] = {0};

    fptr = fopen(ref_file, "r");
    fgets(ref, MAX_REF_LEN, fptr);
    int ref_len = strlen(ref);
    fclose(fptr);

    char reads[MAX_READ * MAX_READ_LEN] = {0};
    char read[MAX_READ_LEN] = {0};

    fptr = fopen(reads_file, "r");

    int read_count = 0;
    int read_len;

    while (fgets(read, MAX_READ_LEN, fptr))
    {
        read_len = strlen(read);
        if (read[read_len - 1] == '\n')
            read[--read_len] = '\0';

        strcpy(&reads[read_count * MAX_READ_LEN], read);
        read_count++;
    }
    fclose(fptr);

    int kmer_per_read = read_len - k + 1;
    int total_kmer = kmer_per_read * read_count;

    int *counts = calloc(read_count, sizeof(int));

    omp_set_num_threads(THREADS);

    double start = omp_get_wtime() * 1000000;
    kmer_cpu_parallel(ref, reads, counts, ref_len, read_len, MAX_READ_LEN, read_count, k);
    double stop = omp_get_wtime() * 1000000;
    double time = stop - start;
    printf("Time taken by cpu (Parallel) function: %f microseconds\n", time);

    fptr = fopen(out_file, "w");

    fprintf(fptr, "CPU (OpenMP) Results:\n");
    for (int i = 0; i < read_count; i++)
    {
        fprintf(fptr, "%d\n", counts[i]);
    }
    fclose(fptr);
}