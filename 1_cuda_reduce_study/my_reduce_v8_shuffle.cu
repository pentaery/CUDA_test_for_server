#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define THREAD_PER_BLOCK 256

template <unsigned int NUM_PER_BLOCK, unsigned int NUM_PER_THREAD>
__global__ void reduce(float *d_input, float *d_output) {
  float sum = 0.f;
  float *begin = &d_input[blockIdx.x * NUM_PER_BLOCK];

  for (int i = 0; i < NUM_PER_THREAD; ++i) {
    sum += begin[threadIdx.x + i * THREAD_PER_BLOCK];
  }

  sum += __shfl_down_sync(0xffffffff, sum, 16);
  sum += __shfl_down_sync(0xffffffff, sum, 8);
  sum += __shfl_down_sync(0xffffffff, sum, 4);
  sum += __shfl_down_sync(0xffffffff, sum, 2);
  sum += __shfl_down_sync(0xffffffff, sum, 1);

  __shared__ float warpLevelSums[32];

  const int laneId = threadIdx.x % 32;
  const int warpId = threadIdx.x / 32;

  if (laneId == 0) {
    warpLevelSums[warpId] = sum;
  }
  __syncthreads();

  if (warpId == 0) {
    sum = (laneId < blockDim.x / 32) ? warpLevelSums[laneId] : 0.f;
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
  }

  if (threadIdx.x == 0) {
    d_output[blockIdx.x] = sum;
  } 

}

bool check(float *output, float *result, int block_num) {
  for (int i = 0; i < block_num; ++i) {
    if (abs(output[i] - result[i]) > 5e-3) {
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

  constexpr int block_num = 1024;
  constexpr int num_per_block = N / block_num;
  constexpr int num_per_thread = num_per_block / THREAD_PER_BLOCK;
  float *output = (float *)malloc(block_num * sizeof(float));
  float *d_output;
  cudaMalloc((void **)&d_output, block_num * sizeof(float));
  float *result = (float *)malloc(block_num * sizeof(float));

  for (int i = 0; i < N; ++i) {
    input[i] = 2.0 * (float)drand48() - 1.0;
  }

  for (int i = 0; i < block_num; ++i) {
    float cur = 0;
    for (int j = 0; j < num_per_block; ++j) {
      cur += input[i * num_per_block + j];
    }
    result[i] = cur;
  }

  cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 Grid(block_num, 1);
  dim3 Block(THREAD_PER_BLOCK, 1);

  reduce<num_per_block, num_per_thread><<<Grid, Block>>>(d_input, d_output);

  cudaMemcpy(output, d_output, block_num * sizeof(float),
             cudaMemcpyDeviceToHost);

  if (check(output, result, block_num)) {
    printf("Success!\n");
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