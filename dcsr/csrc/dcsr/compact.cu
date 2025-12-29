#include "dcsr.hpp"
#include "../utils/cuda-utils.hpp"
#include <cub/cub.cuh>

__global__ void compact_copy_kernel(
    const uint8_t* const* chunks,
    const uint32_t* old_chunk_ids,
    const uint32_t* old_chunk_offsets,
    const uint32_t* node_sizes,
    const uint32_t* new_offsets,
    uint8_t* new_chunk,
    uint32_t num_nodes
) {
    uint32_t node_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_id >= num_nodes) return;

    uint32_t size = node_sizes[node_id];
    if (size == 0) return;

    uint32_t old_chunk_id = old_chunk_ids[node_id];
    uint32_t old_offset = old_chunk_offsets[node_id];
    uint32_t new_offset = new_offsets[node_id];

    const uint8_t* src = chunks[old_chunk_id] + old_offset;
    uint8_t* dst = new_chunk + new_offset;

    for (uint32_t i = 0; i < size; ++i) {
        dst[i] = src[i];
    }
}

__global__ void set_chunk_id_kernel(
    uint32_t* chunk_ids, 
    const uint32_t* sizes, 
    uint32_t num_nodes
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;
    chunk_ids[tid] = (sizes[tid] > 0) ? 0 : UINT32_MAX;
}

void Dcsr::ensure_temp_storage(size_t required_bytes) {
    if (_temp_storage_bytes < required_bytes) {
        if (_d_temp_storage) {
            cudaFreeCheck(_d_temp_storage);
        }
        size_t new_cap = std::max(required_bytes, _temp_storage_bytes * 2);
        _d_temp_storage = (void*)cudaMallocCheck(new_cap);
        _temp_storage_bytes = new_cap;
    }
}

void Dcsr::compact() {
    uint32_t* d_offsets = (uint32_t*)cudaMallocCheck(sizeof(uint32_t) * (_num_nodes + 1));
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes,
        _node_data_size.data_cuda(),
        d_offsets,
        _num_nodes + 1
    );

    ensure_temp_storage(temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(
        _d_temp_storage, temp_storage_bytes,
        _node_data_size.data_cuda(),
        d_offsets,
        _num_nodes + 1
    );

    uint32_t total_size;
    cudaMemcpyD2HCheck(&total_size, d_offsets + _num_nodes, sizeof(uint32_t));

    uint32_t new_chunk_id = _mem_pool.alloc_new_chunk(total_size);
    uint8_t *new_chunk = _mem_pool.get_chunk_device_ptr(new_chunk_id);

    uint32_t threads = 256;
    uint32_t blocks = (_num_nodes + threads - 1) / threads;

    compact_copy_kernel<<<blocks, threads>>>(
        _mem_pool.view().device_chunks,
        _node_chunk_id.data_cuda(),
        _node_chunk_offset.data_cuda(),
        _node_data_size.data_cuda(),
        d_offsets, new_chunk, _num_nodes
    );

    set_chunk_id_kernel<<<blocks, threads>>>(
        _node_chunk_id.data_cuda(),
        _node_data_size.data_cuda(),
        _num_nodes
    );

    cudaMemcpyD2DAsyncCheck(
        _node_chunk_offset.data_cuda(),
        d_offsets,
        sizeof(uint32_t) * _num_nodes
    ); // for those size zero nodes, the offset value doesn't matter

    cudaFreeCheck(d_offsets);

    _mem_pool.reset_to_last();
    // _mem_pool.cpu_chunk(0); // Optional: sync the new chunk to CPU if needed
    _node_chunk_id.cpu_async();
    _node_chunk_offset.cpu_async();
    
    cudaSyncCheck();
}
