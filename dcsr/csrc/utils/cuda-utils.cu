// cuda-utils.cu
#include "cuda-utils.hpp"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

int num_gpus;
thread_local int current_gpu = 0;

void checkCUDA(cudaError_t status) {
  if(status != cudaSuccess) {
    fprintf(stderr, "[ERROR] CUDA Runtime Error: %s\n", cudaGetErrorString(status));
    exit(1);
  }
}

void cudaSetThreadCurrentGPUCheck(int id) {
  if(current_gpu == id) return;
  checkCUDA(cudaSetDevice(id));
  current_gpu = id;
}

bool cudaEnablePeerAccessCheck(int cur, int peer) {
  if(cur == peer) return true;
  
  int canAccessPeer;
  checkCUDA(cudaDeviceCanAccessPeer(&canAccessPeer, cur, peer));
  
  if(!canAccessPeer) {
    printf("[WARN] Cannot access peer GPU %d from GPU %d\n", peer, cur);
    return false;
  }
  
  int old_id = current_gpu;
  cudaSetThreadCurrentGPUCheck(cur);
  checkCUDA(cudaDeviceEnablePeerAccess(peer, 0));
  cudaSetThreadCurrentGPUCheck(old_id);
  
  return true;
}

void cudaUtilsInit() {
  cudaGetDeviceCount(&num_gpus);
  printf("[INFO] Number of GPUs available: %d\n", num_gpus);
  
  if(num_gpus > MAX_GPUS) {
    printf("[WARN] The code is compiled to use up to %d GPUs. Extra GPUs will be ignored.\n", MAX_GPUS);
    num_gpus = MAX_GPUS;
  }

  for(int id = 0; id < num_gpus; ++id) {
    cudaSetThreadCurrentGPUCheck(id);
    cudaFreeCheck(0);
    cudaSyncCheck();
  }

  for(int id = 1; id < num_gpus; ++id) {
    cudaEnablePeerAccessCheck(0, id);
    cudaEnablePeerAccessCheck(id, 0);
  }
}

void *cudaMallocCheck(size_t size) {
  void *ret;
  checkCUDA(cudaMalloc(&ret, size));
  return ret;
}

void cudaMemcpyH2DCheck(void *dst, void *src, size_t size) {
  checkCUDA(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

void cudaMemcpyD2HCheck(void *dst, void *src, size_t size) {
  checkCUDA(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

void cudaMemcpyD2DCheck(void *dst, void *src, size_t size) {
  checkCUDA(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
}

void cudaMemsetCheck(void *p, int v, size_t size) {
  checkCUDA(cudaMemset(p, v, size));
}

void cudaFreeCheck(void *p) {
  checkCUDA(cudaFree(p));
}

void cudaSyncCheck() {
  checkCUDA(cudaDeviceSynchronize());
}

void *cudaHostAllocCheck(size_t size, unsigned int flags) {
  void *ptr;
  checkCUDA(cudaHostAlloc(&ptr, size, flags)); 
  return ptr;
}

void cudaFreeHostCheck(void *p) {
  checkCUDA(cudaFreeHost(p));
}

void cudaStreamCreateCheck(cudaStream_t *pStream) {
  checkCUDA(cudaStreamCreate(pStream));
}

void cudaStreamDestroyCheck(cudaStream_t stream) {
  checkCUDA(cudaStreamDestroy(stream));
}

void cudaStreamSyncCheck(cudaStream_t stream=0) {
  checkCUDA(cudaStreamSynchronize(stream));
}

void cudaMemcpyAsyncCheck(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
  checkCUDA(cudaMemcpyAsync(dst, src, count, kind, stream));
}

void cudaMemcpyH2DAsyncCheck(void *dst, const void *src, size_t size, cudaStream_t stream) {
  checkCUDA(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
}

void cudaMemcpyD2HAsyncCheck(void *dst, const void *src, size_t size, cudaStream_t stream) {
  checkCUDA(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
}

void cudaMemcpyD2DAsyncCheck(void *dst, const void *src, size_t size, cudaStream_t stream) {
  checkCUDA(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream));
}