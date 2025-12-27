// cuda-utils.hpp
#pragma once
#include <cstddef>
#include <cuda_runtime.h>

#define DEBUG_INFO

void *cudaMallocCheck(size_t size);
void cudaMemcpyH2DCheck(void *dst, void *src, size_t size);
void cudaMemcpyD2HCheck(void *dst, void *src, size_t size);
void cudaMemcpyD2DCheck(void *dst, void *src, size_t size);
void cudaMemsetCheck(void *p, int v, size_t size);
void cudaFreeCheck(void *p);
void cudaSyncCheck();

const int MAX_GPUS = 4;
extern int num_gpus;
extern thread_local int current_gpu;

void cudaUtilsInit();
void cudaSetThreadCurrentGPUCheck(int id);
bool cudaEnablePeerAccessCheck(int cur, int peer);

void *cudaHostAllocCheck(size_t size, unsigned int flags = cudaHostAllocDefault);
void cudaFreeHostCheck(void *p);
void cudaStreamCreateCheck(cudaStream_t *pStream);
void cudaStreamDestroyCheck(cudaStream_t stream);
void cudaStreamSyncCheck(cudaStream_t stream);
void cudaMemcpyAsyncCheck(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0);
void cudaMemcpyH2DAsyncCheck(void *dst, const void *src, size_t size, cudaStream_t stream = 0);
void cudaMemcpyD2HAsyncCheck(void *dst, const void *src, size_t size, cudaStream_t stream = 0);
void cudaMemcpyD2DAsyncCheck(void *dst, const void *src, size_t size, cudaStream_t stream = 0);