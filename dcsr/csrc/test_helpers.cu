// test_helpers.cu - CUDA kernels for testing DCSR access patterns
#include <cstdint>
#include <stdio.h>
#include "utils/cuda-utils.hpp"

__global__ void verify_dcsr_sum_kernel(
    uint32_t num_nodes,
    const int* d_data,
    const usize* d_start,
    const usize* d_size,
    int* d_results
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    int sum = 0;
    usize start = d_start[tid];
    usize size = d_size[tid];

    for (usize i = 0; i < size; ++i) {
        sum += d_data[start + i];
    }
    d_results[tid] = sum;
}

__global__ void verify_naive_csr_sum_kernel(
    const uint32_t* d_indptr,
    const int* d_data,
    int* d_results,
    uint32_t num_nodes
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    int sum = 0;
    uint32_t start = d_indptr[tid];
    uint32_t end = d_indptr[tid + 1];

    for (uint32_t i = start; i < end; ++i) {
        sum += d_data[i];
    }
    d_results[tid] = sum;
}

extern "C" {
    void dcsr_test_verify_sum(
        uint32_t num_nodes,
        const int* d_data,
        const usize* d_start,
        const usize* d_size,
        int* host_results,
        uint32_t count
    ) {
        int* d_results = (int *)cudaMallocCheck(count * sizeof(int));

        uint32_t threads = 256;
        uint32_t blocks = (count + threads - 1) / threads;

        verify_dcsr_sum_kernel<<<blocks, threads>>>(num_nodes, d_data, d_start, d_size, d_results);
        cudaSyncCheck();

        cudaMemcpyD2HCheck(host_results, d_results, count * sizeof(int));
        cudaFreeCheck(d_results);
    }

    void naive_csr_test_verify_sum(
        const uint32_t* d_indptr,
        const int* d_data,
        int* host_results,
        uint32_t count
    ) {
        int* d_results = (int *)cudaMallocCheck(count * sizeof(int));

        uint32_t threads = 256;
        uint32_t blocks = (count + threads - 1) / threads;

        verify_naive_csr_sum_kernel<<<blocks, threads>>>(d_indptr, d_data, d_results, count);
        cudaSyncCheck();

        cudaMemcpyD2HCheck(host_results, d_results, count * sizeof(int));
        cudaFreeCheck(d_results);
    }
}