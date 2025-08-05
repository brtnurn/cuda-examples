#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define M 512
#define K 128
#define N 256
#define BLOCK_SIZE 32

void cpu_matrix_mult(int *a, int *b, int *c, int m, int k, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int sum = 0;
            for (int l = 0; l < k; l++)
            {
                sum = sum + a[i * k + l] * b[n * l + j];
            }
            c[i * n + j] = sum;
        }
    }
}

__global__ void gpu_matrix_mult(int *a, int *b, int *c, int m, int k, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n)
    {
        int sum = 0;
        for (int i = 0; i < k; i++)
        {
            sum = sum + a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

void init_matrix(int *matrix, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
    {
        matrix[i] = rand() % 10000;
    }
}

bool compare_results(int *a, int *b, int m, int n)
{
    for (int i = 0; i < m * n; i++)
    {
        if (fabs(a[i] - b[i]) > 1e-5)
        {
            return false;
        }
    }
    return true;
}

int main()
{

    int *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    int size_a = sizeof(int) * M * K;
    int size_b = sizeof(int) * K * N;
    int size_c = sizeof(int) * M * N;

    h_a = (int *)malloc(size_a);
    h_b = (int *)malloc(size_b);
    h_c_cpu = (int *)malloc(size_c);
    h_c_gpu = (int *)malloc(size_c);

    srand(time(NULL));
    init_matrix(h_a, M, K);
    init_matrix(h_b, K, N);

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, sizeof(int) * M * K);
    cudaMalloc(&d_b, sizeof(int) * K * N);
    cudaMalloc(&d_c, sizeof(int) * M * N);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c_gpu, size_c, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    auto start = high_resolution_clock::now();
    gpu_matrix_mult<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, K, N);
    cudaDeviceSynchronize();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by gpu function: " << duration.count() << " microseconds" << endl;

    start = high_resolution_clock::now();
    cpu_matrix_mult(h_a, h_b, h_c_cpu, M, K, N);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by cpu function: " << duration.count() << " microseconds" << endl;

    cudaMemcpy(h_c_gpu, d_c, size_c, cudaMemcpyDeviceToHost);

    bool results = compare_results(h_c_cpu, h_c_gpu, M, N);
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