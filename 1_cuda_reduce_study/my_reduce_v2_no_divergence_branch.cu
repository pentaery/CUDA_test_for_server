#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <chrono>

#define THREAD_PER_BLOCK 256

__global__ void reduce(float *d_input, float *d_output) {
  __shared__ float shared[THREAD_PER_BLOCK];
  float *begin = &d_input[blockIdx.x * blockDim.x];
  shared[threadIdx.x] = begin[threadIdx.x];
  __syncthreads();
  for (int i = 1; i < blockDim.x; i *= 2) {
    if (threadIdx.x * (2 * i) < blockDim.x) {
      int index = threadIdx.x * (2 * i);
      shared[index] += shared[index + i];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    d_output[blockIdx.x] = shared[0];
  }
}

bool check(float *output, float *result, int block_num) {
  for (int i = 0; i < block_num; ++i) {
    if (abs(output[i] - result[i]) > 1e-4) {
      return false;
    }
  }
  return true;
}

int main() {

  const int N = 32 * 1024 * 1024;
  float *input = (float *)malloc(N * sizeof(float));
  float *d_input;
  cudaMalloc((void **)&d_input, N * sizeof(float));

  int block_num = N / THREAD_PER_BLOCK;
  float *output = (float *)malloc(block_num * sizeof(float));
  float *d_output;
  cudaMalloc((void **)&d_output, block_num * sizeof(float));
  float *result = (float *)malloc(block_num * sizeof(float));

  for (int i = 0; i < N; ++i) {
    input[i] = 2.0 * (float)drand48() - 1.0;
  }

  for (int i = 0; i < block_num; ++i) {
    float cur = 0;
    for (int j = 0; j < THREAD_PER_BLOCK; ++j) {
      cur += input[i * THREAD_PER_BLOCK + j];
    }
    result[i] = cur;
  }

  cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 Grid(N / THREAD_PER_BLOCK, 1);
  dim3 Block(THREAD_PER_BLOCK, 1);
  auto begin = std::chrono::high_resolution_clock::now();
  reduce<<<Grid, Block>>>(d_input, d_output);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  printf("Time measured: %.7f seconds.\n", elapsed.count() * 1e-9);
  cudaMemcpy(output, d_output, block_num * sizeof(float),
             cudaMemcpyDeviceToHost);

  if (check(output, result, block_num)) {
    printf("Success\n");
  } else {
    printf("Failed\n");
    for (int i = 0; i < block_num; ++i) {
      printf("%lf\n", output[i] - result[i]);
    }
  }

  cudaFree(d_input);
  cudaFree(d_output);
  return 0;
}