#pragma once
#include "dcsr_view.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Dcsr Dcsr_t;

void* dcsr_create_from_bulk(
    uint32_t num_nodes,
    size_t element_size,
    const void** ptrs,
    const uint32_t* sizes,
    size_t count
);

void dcsr_update_sparse(
    void* handle,
    const uint32_t* ids,
    const void** ptrs,
    const uint32_t* sizes,
    size_t count
);

void dcsr_destroy(void* handle);
DcsrView dcsr_get_view(void* handle);

#ifdef __cplusplus
}
#endif
