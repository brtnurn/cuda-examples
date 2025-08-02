#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define SIZE 100000000
#define BLOCK_SIZE 256

__global__ void gpu_vector_sum(int *a, int *b, int *c, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n)
    {
        c[id] = a[id] + b[id];
    }
}

void cpu_vector_sum(int *a, int *b, int *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

void init_vector(int *arr, int n)
{
    for (int i = 0; i < n; i++)
    {
        arr[i] = rand() % 1000;
    }
}

bool compare_results(int *a, int *b, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (a[i] != b[i])
        {
            return false;
        }
    }
    return true;
}

int main()
{

    int *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    size_t size = SIZE * sizeof(int);

    h_a = (int *)malloc(size);
    h_b = (int *)malloc(size);
    h_c_cpu = (int *)malloc(size);
    h_c_gpu = (int *)malloc(size);

    srand(time(NULL));
    init_vector(h_a, SIZE);
    init_vector(h_b, SIZE);

    int *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c_gpu, size, cudaMemcpyHostToDevice);

    int num_blocks = (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto start = high_resolution_clock::now();
    gpu_vector_sum<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, SIZE);
    cudaDeviceSynchronize();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by gpu function: " << duration.count() << " microseconds" << endl;

    start = high_resolution_clock::now();
    cpu_vector_sum(h_a, h_b, h_c_cpu, SIZE);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by cpu function: " << duration.count() << " microseconds" << endl;

    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    bool results = compare_results(h_c_cpu, h_c_gpu, SIZE);
    cout << "Results: " << results << endl;

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}