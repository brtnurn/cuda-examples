#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cassert>
#include <omp.h>

using namespace std;
using namespace std::chrono;

#define MAX_REF_LEN 1000000
#define MAX_READ_LEN 200
#define MAX_READ 20480

extern __global__ void kmer_parallel(char *ref, char *reads, int *counts, int ref_len, int read_len, int max_read_len, int read_count, int k);
extern void kmer_serial(char *ref, char *reads, int *counts, int ref_len, int read_len, int max_read_len, int read_count, int k);
extern void kmer_serial_bm(char *ref, char *reads, int *counts, int ref_len, int read_len, int max_read_len, int read_count, int k);
extern void kmer_cpu_parallel(char *ref, char *reads, int *counts, int ref_len, int read_len, int max_read_len, int read_count, int k);

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        cout << "Incorrect usage" << endl;
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

    int *h_counts_cpu, *h_counts_cpu_bm, *h_counts_gpu, *h_counts_cpu_omp;

    h_counts_cpu = (int *)malloc(read_count * sizeof(int));
    h_counts_cpu_bm = (int *)malloc(read_count * sizeof(int));
    h_counts_gpu = (int *)malloc(read_count * sizeof(int));
    h_counts_cpu_omp = (int *)malloc(read_count * sizeof(int));

    memset(h_counts_cpu, 0, read_count * sizeof(int));
    memset(h_counts_cpu_bm, 0, read_count * sizeof(int));
    memset(h_counts_gpu, 0, read_count * sizeof(int));
    memset(h_counts_cpu_omp, 0, read_count * sizeof(int));

    char *d_ref;
    char *d_reads;
    int *d_counts;

    cudaMalloc(&d_ref, ref_len * sizeof(char));
    cudaMalloc(&d_reads, read_count * MAX_READ_LEN * sizeof(char));
    cudaMalloc(&d_counts, read_count * sizeof(int));

    cudaMemset(d_counts, 0, read_count * sizeof(int));

    cudaMemcpy(d_ref, ref, ref_len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_reads, reads, read_count * MAX_READ_LEN * sizeof(char), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (total_kmer + threads - 1) / threads;

    auto start = high_resolution_clock::now();
    kmer_serial(ref, reads, h_counts_cpu, ref_len, read_len, MAX_READ_LEN, read_count, k);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by cpu (Naive) function: " << duration.count() << " microseconds" << endl;

    start = high_resolution_clock::now();
    kmer_serial_bm(ref, reads, h_counts_cpu_bm, ref_len, read_len, MAX_READ_LEN, read_count, k);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by cpu (Boyer-Moore) function: " << duration.count() << " microseconds" << endl;

    start = high_resolution_clock::now();
    kmer_parallel<<<blocks, threads>>>(d_ref, d_reads, d_counts, ref_len, read_len, MAX_READ_LEN, read_count, k);
    cudaDeviceSynchronize();
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by gpu function: " << duration.count() << " microseconds" << endl;

    // omp_set_num_threads(8);

    start = high_resolution_clock::now();
    kmer_cpu_parallel(ref, reads, h_counts_cpu_omp, ref_len, read_len, MAX_READ_LEN, read_count, k);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by cpu (Parallel) function: " << duration.count() << " microseconds" << endl;

    // char *result_reads = (char *)malloc(read_count * sizeof(char));

    // cudaMemcpy(result_reads, d_reads, read_count * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_counts_gpu, d_counts, read_count * sizeof(int), cudaMemcpyDeviceToHost);

    fptr = fopen(out_file, "w");

    fprintf(fptr, "CPU (Naive) Results:\n");
    for (int i = 0; i < read_count; i++)
    {
        fprintf(fptr, "%d\n", h_counts_cpu[i]);
    }

    fprintf(fptr, "CPU (Boyer-Moore) Results:\n");
    for (int i = 0; i < read_count; i++)
    {
        fprintf(fptr, "%d\n", h_counts_cpu_bm[i]);
    }

    fprintf(fptr, "GPU Results:\n");
    for (int i = 0; i < read_count; i++)
    {
        fprintf(fptr, "%d\n", h_counts_gpu[i]);
    }

    fprintf(fptr, "CPU (Parallel) Results:\n");
    for (int i = 0; i < read_count; i++)
    {
        fprintf(fptr, "%d\n", h_counts_cpu_omp[i]);
    }
    fclose(fptr);

    return 0;
}