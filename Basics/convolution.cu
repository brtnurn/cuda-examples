#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cassert>

using namespace std;
using namespace chrono;

#define N 1000000
#define THREADS 256
#define MASK_SIZE 16

__constant__ int MASK[MASK_SIZE];

__global__ void gpu_convolution(int *array, int *result)
{
    extern __shared__ int shmem[];

    int padded = THREADS + MASK_SIZE / 2 * 2;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int offset = threadIdx.x + blockDim.x;

    shmem[threadIdx.x] = array[tid];

    if (offset < padded)
    {
        shmem[offset] = array[tid + offset];
    }
    __syncthreads();

    int temp = 0;
    for (int i = 0; i < MASK_SIZE; i++)
    {
        temp += shmem[threadIdx.x + i] * MASK[i];
    }

    result[tid] = temp;
}

void cpu_convolution(int *array, int *mask, int *result, int n)
{
    int temp;
    for (int i = 0; i < n; i++)
    {
        temp = 0;
        for (int j = 0; j < MASK_SIZE; j++)
        {
            temp += array[i + j] * mask[j];
        }
    }
}

void init_array(int *arr, int n, int r)
{
    for (int i = 0; i < n; i++)
    {
        if (i < r || i >= n - r)
        {
            arr[i] = 0;
        }
        else
        {
            arr[i] = rand() % 100;
        }
    }
}

int main()
{
    int r = MASK_SIZE / 2;
    int p_array_size = N + r * 2;

    int *h_array, *h_result_cpu, *h_result_gpu, *h_mask;

    h_array = (int *)malloc(p_array_size * sizeof(int));
    h_result_cpu = (int *)malloc(N * sizeof(int));
    h_result_gpu = (int *)malloc(N * sizeof(int));
    h_mask = (int *)malloc(MASK_SIZE * sizeof(int));

    srand(time(NULL));
    init_array(h_array, p_array_size, r);
    for (int i = 0; i < MASK_SIZE; i++)
    {
        h_mask[i] = rand() % 10;
    }

    int *d_array, *d_result;
    cudaMalloc(&d_array, p_array_size * sizeof(int));
    cudaMalloc(&d_result, N * sizeof(int));

    cudaMemcpy(d_array, h_array, p_array_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(MASK, h_mask, MASK_SIZE * sizeof(int));

    int blocks = (N + THREADS - 1) / THREADS;

    int shmem = THREADS + r * 2;

    auto start = high_resolution_clock::now();
    gpu_convolution<<<blocks, THREADS, shmem>>>(d_array, d_result);
    cudaDeviceSynchronize();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by gpu function: " << duration.count() << " microseconds" << endl;

    cudaMemcpy(h_result_gpu, d_result, N * sizeof(int), cudaMemcpyDeviceToHost);

    start = high_resolution_clock::now();
    cpu_convolution(h_array, h_mask, h_result_cpu, N);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by cpu function: " << duration.count() << " microseconds" << endl;

    cout << "COMPLETED" << endl;

    return 0;
}