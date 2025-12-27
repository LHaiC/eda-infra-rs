// mempool.hpp
#pragma once
#include "cuda-utils.hpp"
#include "heteromem.hpp"
#include <vector>
#include <cstdint>
#include <numeric>
#include <cstdio>

// Use PinnedAllocator for optimal Host-to-Device transfer performance
using DefaultAllocator = PinnedAllocator;

class HeteroMemPool {
private:
    std::vector<uint8_t*> _host_chunks;                           // Host memory blocks
    HeteroMem<uint8_t*, DefaultAllocator> _hetero_chunks;         // Array of device pointers synchronized to GPU
    std::vector<size_t> _chunk_sizes;                             // Size of each chunk in bytes
    size_t _total_chunks_used = 0;                                // Total number of chunks currently used
    size_t _total_bytes_used = 0;                                 // Total bytes used across all chunks
    size_t _num_chunks_reserved = 8;                              // Number of chunks reserved

public:
    HeteroMemPool() = default;
    
    ~HeteroMemPool() {
        for (uint8_t* ptr : _host_chunks) {
            DefaultAllocator::deallocate(ptr);
        }
        // Manually free device memory blocks tracked by HeteroMem
        for (size_t i = 0; i < _hetero_chunks.size(); ++i) {
            if (_hetero_chunks[i]) {
                cudaFreeCheck(_hetero_chunks[i]);
            }
        }
    }

    // Disable copy to prevent resource management conflicts
    HeteroMemPool(const HeteroMemPool&) = delete;
    HeteroMemPool& operator=(const HeteroMemPool&) = delete;
    HeteroMemPool(HeteroMemPool&&) = delete;
    HeteroMemPool& operator=(HeteroMemPool&&) = delete;

    uint32_t alloc_new_chunk(size_t bytes_needed, cudaStream_t stream = 0) {
        if (bytes_needed == 0) return UINT32_MAX;

        // Get index for the current request
        uint32_t idx = _total_chunks_used++;
        bool update_gpu = false;

        // Case 1: Expand pool (New allocation)
        if (idx >= _host_chunks.size()) {
            uint8_t* h_ptr = DefaultAllocator::allocate<uint8_t>(bytes_needed);
            uint8_t* d_ptr = (uint8_t*)cudaMallocCheck(bytes_needed);

            _host_chunks.push_back(h_ptr);
            _chunk_sizes.push_back(bytes_needed);
            
            // Resize directory and store device ptr
            _hetero_chunks.resize(idx + 1);
            _hetero_chunks[idx] = d_ptr;
            
            update_gpu = true;
        } 
        // Case 2: Reuse existing slot (Grow if insufficient)
        else if (_chunk_sizes[idx] < bytes_needed) {
            // Free obsolete memory
            DefaultAllocator::deallocate(_host_chunks[idx]);
            cudaFreeCheck(_hetero_chunks[idx]);
            
            _host_chunks[idx] = DefaultAllocator::allocate<uint8_t>(bytes_needed);
            _hetero_chunks[idx] = (uint8_t*)cudaMallocCheck(bytes_needed);
            _chunk_sizes[idx] = bytes_needed;

            update_gpu = true;
        }
        // Case 3: Direct reuse (Size sufficient) -> Fall through

        // Sync metadata only if pointer changed
        if (update_gpu) {
            _hetero_chunks.cuda_async(idx, idx + 1, stream);
        }

        _total_bytes_used += bytes_needed;
        return idx;
    }

    // Logical reset; physical memory is retained for reuse
    void reset() {
        _total_chunks_used = 0;
        _total_bytes_used = 0;
    }

    // Reset to last allocated chunk only
    void reset_to_last(cudaStream_t stream = 0) {
        if (_total_chunks_used <= 1) return;

        uint32_t last_idx = _total_chunks_used - 1;
        uint8_t* last_host_ptr = _host_chunks[last_idx];
        uint8_t* last_device_ptr = _hetero_chunks[last_idx];
        size_t last_size = _chunk_sizes[last_idx];

        size_t keep_count = std::min(_total_chunks_used - 1, _num_chunks_reserved);

        if (_host_chunks[0]) DefaultAllocator::deallocate(_host_chunks[0]);
        if (_hetero_chunks[0]) cudaFreeCheck(_hetero_chunks[0]);
        for (size_t i = keep_count; i < _host_chunks.size(); ++i) {
            if (i == last_idx) continue;
            if (_host_chunks[i]) DefaultAllocator::deallocate(_host_chunks[i]);
            if (_hetero_chunks[i]) cudaFreeCheck(_hetero_chunks[i]);
        }

        _host_chunks[0] = last_host_ptr;
        _hetero_chunks[0] = last_device_ptr;
        _chunk_sizes[0] = last_size;

        _host_chunks.resize(keep_count);
        _hetero_chunks.resize(keep_count);
        _chunk_sizes.resize(keep_count);

        _hetero_chunks.cuda_async(stream);

        _total_chunks_used = 1;
        _total_bytes_used = last_size;
    }

    uint8_t* get_chunk_host_ptr(uint32_t chunk_idx) {
        return _host_chunks[chunk_idx];
    }

    uint8_t* get_chunk_device_ptr(uint32_t chunk_idx) {
        return _hetero_chunks[chunk_idx];
    }
    
    void cuda_chunk(uint32_t chunk_idx, cudaStream_t stream = 0) {
        if (_chunk_sizes[chunk_idx] > 0) {
            cudaMemcpyH2DAsyncCheck(
                _hetero_chunks[chunk_idx],
                _host_chunks[chunk_idx],
                _chunk_sizes[chunk_idx],
                stream
            );
        }
    }

    void cuda_all(cudaStream_t stream = 0) {
        for (size_t i = 0; i < _total_chunks_used; ++i) {
            cuda_chunk(i, stream);
        }
    }

    void cpu_chunk(uint32_t chunk_idx, cudaStream_t stream = 0) {
        if (_chunk_sizes[chunk_idx] > 0) {
            cudaMemcpyD2HAsyncCheck(
                _host_chunks[chunk_idx],
                _hetero_chunks[chunk_idx],
                _chunk_sizes[chunk_idx],
                stream
            );
        }
    }

    void cpu_all(cudaStream_t stream = 0) {
        for (size_t i = 0; i < _total_chunks_used; ++i) {
            cpu_chunk(i, stream);
        }
    }

    size_t total_chunks_used() const {
        return _total_chunks_used;
    }

    size_t total_bytes_used() const {
        return _total_bytes_used;
    }
    
    // View structure for accessing memory chunks inside CUDA kernels
    struct DeviceView {
        uint8_t* const* device_chunks; // Pointer to the device pointer array (T**) on GPU
    };

    DeviceView view() const {
        return { const_cast<uint8_t* const*>(_hetero_chunks.data_cuda()) };
    }
};