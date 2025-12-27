#include "dcsr.hpp"
#include "dcsr_capi.h"
#include <stdexcept>
#include <omp.h>

namespace {
    template<typename T>
    void sort_and_unique(std::vector<T>& vec) {
        if (vec.empty()) return;
        std::sort(vec.begin(), vec.end());
        vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
    }
}

Dcsr::Dcsr(uint32_t num_nodes, uint32_t element_size)
        : _element_size(element_size) {
    if (element_size == 0) throw std::invalid_argument("Element size cannot be zero.");
    resize_state(num_nodes);
}

Dcsr::~Dcsr() {
    if (_d_temp_storage) {
        cudaFreeCheck(_d_temp_storage);
        _d_temp_storage = nullptr;
    }
}

void Dcsr::resize_state(uint32_t num_nodes) {
    uint32_t old_num_nodes = _num_nodes;
    _num_nodes = num_nodes;
    _node_chunk_id.resize(num_nodes);
    _node_chunk_offset.resize(num_nodes);
    _node_data_size.resize(num_nodes);

    for (uint32_t i = old_num_nodes; i < num_nodes; ++i) {
        _node_chunk_id[i] = UINT32_MAX;
        _node_chunk_offset[i] = 0;
        _node_data_size[i] = 0;
    }
}

void Dcsr::bulk_load(const void** ptrs, const uint32_t* sizes, size_t count) {
    if (count > _num_nodes) resize_state(count);

    // Phase 1: Parallel prefix sum to compute offsets
    std::vector<uint32_t> offsets(count);
    size_t total_bytes = 0;

#pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
        uint32_t size = sizes[i];
        _node_data_size[i] = size;
        offsets[i] = size;
    }

    // Convert sizes to offsets using exclusive scan
    uint32_t running_offset = 0;
    for (size_t i = 0; i < count; ++i) {
        uint32_t size = offsets[i];
        offsets[i] = running_offset;
        running_offset += size;
    }
    total_bytes = running_offset;

    // Phase 2: Allocate chunk and copy data directly (parallel)
    if (total_bytes > 0) {
        uint32_t new_chunk_id = _mem_pool.alloc_new_chunk(total_bytes);
        uint8_t* chunk_ptr = _mem_pool.get_chunk_host_ptr(new_chunk_id);

#pragma omp parallel for
        for (size_t i = 0; i < count; ++i) {
            uint32_t size = sizes[i];
            if (size == 0) continue;

            // Copy to final chunk
            _node_chunk_id[i] = new_chunk_id;
            _node_chunk_offset[i] = offsets[i];
            std::memcpy(chunk_ptr + offsets[i], ptrs[i], size);
        }
    } else {
#pragma omp parallel for
        for (size_t i = 0; i < count; ++i) {
            _node_chunk_id[i] = UINT32_MAX;
            _node_chunk_offset[i] = 0;
        }
    }

    _total_data_bytes = total_bytes;
    sync_all();
}

void Dcsr::update_sparse(const uint32_t* ids, const void** ptrs, const uint32_t* sizes, size_t count) {
    if (count == 0) return;
    uint32_t num_nodes = _num_nodes;
    size_t total_bytes = 0;
    for (size_t i = 0; i < count; ++i) {
        uint32_t u = ids[i];
        if (u >= num_nodes) num_nodes = u + 1;
        total_bytes += sizes[i];
    }
    if (num_nodes > _num_nodes) resize_state(num_nodes);
    uint32_t new_chunk_id = _mem_pool.alloc_new_chunk(total_bytes);
    uint8_t* chunk_ptr = _mem_pool.get_chunk_host_ptr(new_chunk_id);
    uint32_t current_offset = 0;

    for (size_t i = 0; i < count; ++i) {
        uint32_t u = ids[i];
        uint32_t size = sizes[i];
        uint32_t old_size = _node_data_size[u];
        _total_data_bytes += (size - old_size);

        if (size == 0) {
            _node_data_size[u] = 0;
            _node_chunk_id[u] = UINT32_MAX;
            _node_chunk_offset[u] = 0;
            continue;
        }
        _node_data_size[u] = size;
        _node_chunk_id[u] = new_chunk_id;
        _node_chunk_offset[u] = current_offset;
        // Zero-copy: directly copy from Rust's pinned memory
        std::memcpy(chunk_ptr + current_offset, ptrs[i], size);
        current_offset += size;
    }

    _mem_pool.cuda_chunk(new_chunk_id);
    _node_data_size.cuda_async();
    _node_chunk_id.cuda_async();
    _node_chunk_offset.cuda_async();

    if (needs_compaction()) {
        compact();
    }
    cudaSyncCheck();
}

bool Dcsr::needs_compaction() const {
    size_t total_capacity = _mem_pool.total_bytes_used();
    if (total_capacity == 0) return false;
    return (double)_total_data_bytes / total_capacity < _compaction_threshold;
}

void Dcsr::sync_all() {
    _mem_pool.cuda_all();
    _node_chunk_id.cuda_async();
    _node_chunk_offset.cuda_async();
    _node_data_size.cuda_async();
    cudaSyncCheck();
}

DcsrView Dcsr::device_view() const {
    return DcsrView {
        _node_chunk_id.data_cuda(),
        _node_chunk_offset.data_cuda(),
        _node_data_size.data_cuda(),
        const_cast<uint8_t* const*>(_mem_pool.view().device_chunks),
        _num_nodes,
        _element_size
    };
}

extern "C" {

void* dcsr_create_from_bulk(
    uint32_t num_nodes,
    size_t element_size,
    const void** ptrs,
    const uint32_t* sizes,
    size_t count
) {
    try {
        auto* dcsr = new Dcsr(num_nodes, static_cast<uint32_t>(element_size));
        dcsr->bulk_load(ptrs, sizes, count);
        return dcsr;
    } catch (...) {
        return nullptr;
    }
}

void dcsr_update_sparse(
    void* handle,
    const uint32_t* ids,
    const void** ptrs,
    const uint32_t* sizes,
    size_t count
) {
    if (!handle || !ids || !ptrs || !sizes || count == 0) return;
    auto* dcsr = reinterpret_cast<Dcsr*>(handle);
    dcsr->update_sparse(ids, ptrs, sizes, count);
}

void dcsr_destroy(void* handle) {
    delete reinterpret_cast<Dcsr*>(handle);
}

DcsrView dcsr_get_view(void* handle) {
    if (!handle) {
        return DcsrView{};
    }
    return reinterpret_cast<Dcsr*>(handle)->device_view();
}

}
