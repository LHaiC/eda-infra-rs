#pragma once
#include "dcsr_view.h"

namespace dcsr {

    template <typename T>
    struct NodeSpan {
    private:
        const DcsrView& _view;
        uint32_t _node_id;
        uint32_t _size;

        const T* _base_ptr;
        bool _is_valid;

    public:
        __device__ __forceinline__ NodeSpan(const DcsrView& v, uint32_t id)
            : _view(v), _node_id(id), _base_ptr(nullptr), _is_valid(false)
        {
            _size = v.node_data_size_bytes[id] / sizeof(T);
            uint32_t chunk_id = v.node_chunk_id[id];
            uint32_t chunk_offset = v.node_chunk_offset[id];
            _is_valid = (chunk_id != UINT32_MAX);
            if (_is_valid) {
                const uint8_t* ptr = v.get_ptr(chunk_id, chunk_offset);
                _base_ptr = reinterpret_cast<const T*>(ptr);
            }
        }

        __device__ __forceinline__ T operator[](uint32_t i) const {
            if (!_is_valid || i >= _size) {
                return T{};
            }
            return _base_ptr[i];
        }

        __device__ __forceinline__ uint32_t size() const { return _size; }
        __device__ __forceinline__ bool empty() const { return _size == 0; }

        struct Iterator {
            const NodeSpan<T>& span;
            uint32_t idx;

            __device__ __forceinline__ bool operator!=(const Iterator& other) const { return idx != other.idx; }
            __device__ __forceinline__ void operator++() { idx++; }
            __device__ __forceinline__ T operator*() const { return span[idx]; }
        };

        __device__ __forceinline__ Iterator begin() const { return Iterator{*this, 0}; }
        __device__ __forceinline__ Iterator end() const { return Iterator{*this, _size}; }
    };

    template <typename T>
    __device__ __forceinline__ NodeSpan<T> neighbors(const DcsrView& view, uint32_t node_id) {
        return NodeSpan<T>(view, node_id);
    }

} // namespace dcsr