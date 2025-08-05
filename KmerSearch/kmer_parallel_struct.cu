#include <cuda_runtime.h>
#include <iostream>

using namespace std;

#define MAX_REF_LEN 100
#define MAX_READ_LEN 20
#define MAX_READ 20

typedef struct kmer
{
    char str[MAX_READ_LEN];
    int count;
    int hits[MAX_REF_LEN];
} kmer;

typedef struct read
{
    char str[MAX_READ_LEN];
    int count;
    kmer *kmers;

} read;

__global__ void kmer_parallel(char *ref, read *reads, int ref_len, int read_len, int read_count, int k)
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
        return; // boundary check

    int count = 0;

    for (int i = 0; i < ref_len - k + 1; i++)
    {
        bool flag = true;
        for (int j = 0; j < k; j++)
        {
            if (ref[i + j] != reads[read_idx].kmers[kmer_idx].str[j])
            {
                flag = false;
                break;
            }
        }
        if (flag)
        {
            reads[read_idx].kmers[kmer_idx].hits[count++] = i;
        }
    }
    atomicAdd(&reads[read_idx].count, count);
    reads[read_idx].kmers[kmer_idx].count = count;
}

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

    // Read reference
    char ref[MAX_REF_LEN] = {0};

    fptr = fopen(ref_file, "r");
    fgets(ref, MAX_REF_LEN, fptr);
    int ref_len = strlen(ref);
    fclose(fptr);

    // Read reads
    read reads[MAX_READ];
    char read_name[MAX_READ_LEN] = {0};

    fptr = fopen(reads_file, "r");

    int read_count = 0;
    int read_len;
    int kmer_count;

    while (fgets(read_name, MAX_READ_LEN, fptr))
    {
        read_len = strlen(read_name);
        if (read_name[read_len - 1] == '\n')
        {
            read_name[read_len - 1] = '\0';
            read_len--;
        }
        read myRead;
        strcpy(myRead.str, read_name);
        myRead.count = 0;
        kmer_count = read_len - k + 1;
        myRead.kmers = (kmer *)malloc(kmer_count * sizeof(kmer));

        for (int i = 0; i < kmer_count; i++)
        {
            char kmer_name[k + 1];
            for (int j = 0; j < k; j++)
            {
                kmer_name[j] = read_name[i + j];
            }
            kmer_name[k] = '\0';
            kmer myKmer = {0};
            strcpy(myKmer.str, kmer_name);
            myRead.kmers[i] = myKmer;
        }
        reads[read_count++] = myRead;
    }
    fclose(fptr);
    int total_kmer = kmer_count * read_count;

    char *d_ref;
    read *d_reads;

    cudaMalloc(&d_ref, ref_len * sizeof(char));
    cudaMalloc(&d_reads, read_count * sizeof(read));

    cudaMemcpy(d_ref, ref, ref_len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_reads, reads, read_count * sizeof(read), cudaMemcpyHostToDevice);

    for (int i = 0; i < read_count; i++)
    {
        // Allocate kmers array on device
        kmer *d_kmers;
        cudaMalloc(&d_kmers, kmer_count * sizeof(kmer));

        // Copy kmers array to device
        cudaMemcpy(d_kmers, reads[i].kmers, kmer_count * sizeof(kmer), cudaMemcpyHostToDevice);

        // Update device pointer for kmers inside d_reads[i]
        cudaMemcpy(&(d_reads[i].kmers), &d_kmers, sizeof(kmer *), cudaMemcpyHostToDevice);
    }

    int threads = 128;
    int blocks = (total_kmer + threads - 1) / threads;

    kmer_parallel<<<blocks, threads>>>(d_ref, d_reads, ref_len, read_len, read_count, k);
    cudaDeviceSynchronize();

    cudaGetLastError();
    cudaDeviceSynchronize();

    read *result_reads = (read *)malloc(read_count * sizeof(read));
    cudaMemcpy(result_reads, d_reads, read_count * sizeof(read), cudaMemcpyDeviceToHost);

    for (int i = 0; i < read_count; i++)
    {
        // Allocate space for kmers on host
        result_reads[i].kmers = (kmer *)malloc(kmer_count * sizeof(kmer));

        // Copy kmers from device to host
        kmer *d_kmers;
        cudaMemcpy(&d_kmers, &(d_reads[i].kmers), sizeof(kmer *), cudaMemcpyDeviceToHost);
        cudaMemcpy(result_reads[i].kmers, d_kmers, kmer_count * sizeof(kmer), cudaMemcpyDeviceToHost);
    }

    fptr = fopen(out_file, "w");
    for (int i = 0; i < read_count; i++)
    {
        fprintf(fptr, "%d\n", result_reads[i].count);
        cout << "----------- Read " << i << " -----------" << endl;
        for (int j = 0; j < kmer_count; j++)
        {
            kmer kmer = result_reads[i].kmers[j];
            cout << "Kmer " << "(" << kmer.str << "): ";
            for (int k = 0; k < kmer.count; k++)
            {
                cout << kmer.hits[k] << ", ";
            }
            cout << endl;
        }
    }
    fclose(fptr);

    return 0;
}
