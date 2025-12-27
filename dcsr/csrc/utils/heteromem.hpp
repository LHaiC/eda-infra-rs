// heteromem.hpp
#pragma once
#include <cassert>
#include <algorithm>
#include <cstring>
#include <utility>
#include <type_traits>
#include <vector>
#include "cuda-utils.hpp"

struct PageableAllocator {
    template<typename T>
    static T* allocate(size_t n) { 
        if (n == 0) return nullptr;
        return new T[n]; 
    }
    
    template<typename T>
    static void deallocate(T* p) { 
        if (!p) return;
        delete[] p; 
    }
};

struct PinnedAllocator {
    template<typename T>
    static T* allocate(size_t n) {
        if (n == 0) return nullptr;
        return (T*)cudaHostAllocCheck(sizeof(T) * n);
    }
    
    template<typename T>
    static void deallocate(T* p) {
        if (!p) return;
        cudaFreeHostCheck(p);
    }
};

template<typename T, typename Allocator = PageableAllocator, bool InflateOnReserve = true>
struct HeteroMem {
private:
    static constexpr float InflationFactor = 1.5f;
    T* _data_cpu = nullptr;
    T* _data_cuda = nullptr;
    size_t _size = 0;
    size_t _capacity = 0;

public:
    HeteroMem() = default;
    ~HeteroMem() {
        free();
    }

    HeteroMem(const HeteroMem&) = delete;
    HeteroMem& operator=(const HeteroMem&) = delete;
    HeteroMem(HeteroMem&& other) = delete;
    HeteroMem& operator=(HeteroMem&& other) = delete;

    inline void free() {
        if (_data_cpu) {
            Allocator::template deallocate<T>(_data_cpu);
            _data_cpu = nullptr;
        }
        if (_data_cuda) {
            cudaFreeCheck(_data_cuda);
            _data_cuda = nullptr;
        }
        _size = 0;
        _capacity = 0;
    }

    inline void reserve(size_t new_capacity) {
        if (new_capacity <= _capacity) return;

        size_t next_cap = InflateOnReserve ? std::max(new_capacity, (size_t)(_capacity * InflationFactor)) : new_capacity;
        T* new_cpu = Allocator::template allocate<T>(next_cap);
        T* new_gpu = (T*)cudaMallocCheck(sizeof(T) * next_cap);

        if (_size > 0) {
            if (std::is_trivially_copyable<T>::value) {
                std::memcpy(new_cpu, _data_cpu, _size * sizeof(T));
            } else {
                std::move(_data_cpu, _data_cpu + _size, new_cpu);
            }
            if (_data_cuda) {
                cudaMemcpyD2DCheck(new_gpu, _data_cuda, _size * sizeof(T));
            }
        }

        if (_data_cpu) Allocator::template deallocate<T>(_data_cpu);
        if (_data_cuda) cudaFreeCheck(_data_cuda);

        _data_cpu = new_cpu;
        _data_cuda = new_gpu;
        _capacity = next_cap;
    }

    __forceinline__ void resize(size_t new_size) {
        if (new_size > _capacity) {
            reserve(new_size);
        }
        _size = new_size;
    }

    __forceinline__ void allocate(size_t new_size) {
        resize(new_size);
    }

    __forceinline__ void append(size_t additional_size) {
        resize(_size + additional_size);
    }

inline void cuda() {
        if (_size == 0 || _data_cuda == nullptr) return;
        cudaMemcpyH2DCheck(_data_cuda, _data_cpu, _size * sizeof(T));
    }

    inline void cuda(size_t l, size_t r) {
        if (l >= r || _size == 0) return;
        assert(r <= _size);
        cudaMemcpyH2DCheck(_data_cuda + l, _data_cpu + l, (r - l) * sizeof(T));
    }

    inline void cuda_size(size_t l, size_t size) {
        if (size == 0 || _size == 0) return;
        assert(l + size <= _size);
        cudaMemcpyH2DCheck(_data_cuda + l, _data_cpu + l, size * sizeof(T));
    }

    inline void cpu() {
        if (_size == 0) return;
        cudaMemcpyD2HCheck(_data_cpu, _data_cuda, _size * sizeof(T));
    }

    inline void cpu(size_t l, size_t r) {
        if (l >= r || _size == 0) return;
        assert(r <= _size);
        cudaMemcpyD2HCheck(_data_cpu + l, _data_cuda + l, (r - l) * sizeof(T));
    }

    inline void cpu_size(size_t l, size_t size) {
        if (size == 0 || _size == 0) return;
        assert(l + size <= _size);
        cudaMemcpyD2HCheck(_data_cpu + l, _data_cuda + l, size * sizeof(T));
    }

    inline void cuda_async(cudaStream_t stream = 0) {
        if (_size == 0 || _data_cuda == nullptr) return;
        cudaMemcpyH2DAsyncCheck(_data_cuda, _data_cpu, _size * sizeof(T), stream);
    }

    inline void cuda_async(size_t l, size_t r, cudaStream_t stream = 0) {
        if (l >= r || _size == 0) return;
        assert(r <= _size);
        cudaMemcpyH2DAsyncCheck(_data_cuda + l, _data_cpu + l, (r - l) * sizeof(T), stream);
    }

    inline void cuda_async_size(size_t l, size_t size, cudaStream_t stream = 0) {
        if (size == 0 || _size == 0) return;
        assert(l + size <= _size);
        cudaMemcpyH2DAsyncCheck(_data_cuda + l, _data_cpu + l, size * sizeof(T), stream);
    }

    inline void cpu_async(cudaStream_t stream = 0) {
        if (_size == 0) return;
        cudaMemcpyD2HAsyncCheck(_data_cpu, _data_cuda, _size * sizeof(T), stream);
    }

    inline void cpu_async(size_t l, size_t r, cudaStream_t stream = 0) {
        if (l >= r || _size == 0) return;
        assert(r <= _size);
        cudaMemcpyD2HAsyncCheck(_data_cpu + l, _data_cuda + l, (r - l) * sizeof(T), stream);
    }

    inline void cpu_async_size(size_t l, size_t size, cudaStream_t stream = 0) {
        if (size == 0 || _size == 0) return;
        assert(l + size <= _size);
        cudaMemcpyD2HAsyncCheck(_data_cpu + l, _data_cuda + l, size * sizeof(T), stream);
    }

    inline void set_cpu_data(const T* data) {
        if (_size == 0) return;
        if (std::is_trivially_copyable<T>::value) {
            std::memcpy(_data_cpu, data, _size * sizeof(T));
        } else {
            std::copy(data, data + _size, _data_cpu);
        }
    }

    inline void set_cpu_data(const std::vector<T>& data) {
        set_cpu_data(data.data());
    }

    __forceinline__ T* data_cpu() { return _data_cpu; }
    __forceinline__ const T* data_cpu() const { return _data_cpu; }

    __forceinline__ T* data_cuda() { return _data_cuda; }
    __forceinline__ const T* data_cuda() const { return _data_cuda; }

    __forceinline__ size_t size() const { return _size; }
    __forceinline__ size_t capacity() const { return _capacity; }
    __forceinline__ bool empty() const { return _size == 0; }

    __forceinline__ T& operator[](size_t i) {
        assert(i < _size);
        return _data_cpu[i];
    }

    __forceinline__ const T& operator[](size_t i) const {
        assert(i < _size);
        return _data_cpu[i];
    }
};