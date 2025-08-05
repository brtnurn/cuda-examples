#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <cassert>

using namespace std;
using namespace std::chrono;

#define TILE_SIZE 16
#define N 1024

__global__ void gpu_tiled_matrix_mult(int *a, int *b, int *c)
{
    __shared__ int tileA[TILE_SIZE][TILE_SIZE];
    __shared__ int tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tmp = 0;

    for (int i = 0; i < N / TILE_SIZE; i++)
    {
        tileA[threadIdx.y][threadIdx.x] = a[row * N + (i * blockDim.x + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = b[(i * blockDim.y + threadIdx.y) * N + col];

        __syncthreads();

        for (int j = 0; j < TILE_SIZE; j++)
        {
            // tmp += tileA[threadIdx.y * TILE_SIZE + j] * tileB[j * TILE_SIZE + threadIdx.x];
            tmp += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
        }

        __syncthreads();
    }

    c[row * N + col] = tmp;
}

void init_matrix(int *matrix)
{
    for (int i = 0; i < N * N; i++)
    {
        matrix[i] = rand() % 10000;
    }
}

void cpu_tiled_matrix_mult(int *a, int *b, int *c)
{
    // For every row...
    for (int i = 0; i < N; i++)
    {
        // For every column...
        for (int j = 0; j < N; j++)
        {
            // For every element in the row-column pair
            int tmp = 0;
            for (int k = 0; k < N; k++)
            {
                // Accumulate the partial results
                tmp += a[i * N + k] * b[k * N + j];
            }

            // Check against the CPU result
            // assert(tmp == c[i * N + j]);
        }
    }
}

int main()
{

    int *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    size_t size = N * N * sizeof(int);

    h_a = (int *)malloc(size);
    h_b = (int *)malloc(size);
    h_c_cpu = (int *)malloc(size);
    h_c_gpu = (int *)malloc(size);

    srand(time(NULL));
    init_matrix(h_a);
    init_matrix(h_b);

    int *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    auto start = high_resolution_clock::now();
    gpu_tiled_matrix_mult<<<grid_size, block_size>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by gpu function: " << duration.count() << " microseconds" << endl;

    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);

    start = high_resolution_clock::now();
    cpu_tiled_matrix_mult(h_a, h_b, h_c_cpu);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by cpu function: " << duration.count() << " microseconds" << endl;

    cout << "COMPLETED" << endl;

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}