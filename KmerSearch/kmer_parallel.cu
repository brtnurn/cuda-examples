#include <cuda_runtime.h>
#include <iostream>

__global__ void kmer_parallel(char *ref, char *reads, int *counts, int ref_len, int read_len, int max_read_len, int read_count, int k)
{
    /*extern __shared__ char shmem[];

    for (int i = threadIdx.x; i < ref_len; i += blockDim.x)
    {
        shmem[i] = ref[i];
    }
    __syncthreads();*/

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    int kmer_count = read_len - k + 1;

    int read_idx = tid / kmer_count;
    int kmer_idx = tid % kmer_count;

    if (read_idx >= read_count)
    {
        return;
    }

    for (int i = 0; i < ref_len - k + 1; i++)
    {
        bool flag = true;
        for (int j = 0; j < k; j++)
        {
            if (ref[i + j] != reads[read_idx * max_read_len + kmer_idx + j])
            {
                flag = false;
                break;
            }
        }
        if (flag)
        {
            atomicAdd(&counts[read_idx], 1);
        }
    }
}