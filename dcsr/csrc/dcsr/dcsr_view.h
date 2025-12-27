// dcsr_view.h
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

    struct DcsrView {
        const uint32_t* node_chunk_id;
        const uint32_t* node_chunk_offset;
        const uint32_t* node_data_size_bytes;
        uint8_t* const* device_chunks;
        uint32_t num_nodes;
        uint32_t element_size;

#ifdef __cplusplus
        __device__ __forceinline__ const uint8_t* get_ptr(uint32_t chunk_id, uint32_t chunk_offset) const {
            return device_chunks[chunk_id] + chunk_offset;
        }
#endif
    };

#ifdef __cplusplus
}
#endif