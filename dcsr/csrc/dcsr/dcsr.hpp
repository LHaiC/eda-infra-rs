#pragma once
#include "../utils/cuda-utils.hpp"
#include "../utils/heteromem.hpp"
#include "../utils/mempool.hpp"
#include "dcsr_view.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cstdint>
#include <memory>

class Dcsr {
private:
    HeteroMemPool _mem_pool;

    HeteroMem<uint32_t> _node_chunk_id;
    HeteroMem<uint32_t> _node_chunk_offset;
    HeteroMem<uint32_t> _node_data_size;

    uint32_t _num_nodes = 0;
    size_t _total_data_bytes = 0;

    void* _d_temp_storage = nullptr; 
    size_t _temp_storage_bytes = 0;

    float _compaction_threshold = 0.5f;
    const uint32_t _element_size;

    void ensure_temp_storage(size_t required_bytes);
    void compact();
    bool needs_compaction() const;
    void sync_all();

public:
    Dcsr(uint32_t num_nodes, uint32_t element_size);
    ~Dcsr();
    void resize_state(uint32_t num_nodes);

    void bulk_load(const void** ptrs, const uint32_t* sizes, size_t count);
    void update_sparse(const uint32_t* ids, const void** ptrs, const uint32_t* sizes, size_t count);
    DcsrView device_view() const;
};