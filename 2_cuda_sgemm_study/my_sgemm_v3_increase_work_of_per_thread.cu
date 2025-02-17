#include <cstdio>

#define A(i, j) a[i * n + j]
#define B(i, j) b[i * n + j]
void random_matrix(int m, int n, float *a) {
  int i, j;
  for (i = 0; i < m; ++i) {
    for (j = 0; j < n; ++j) {
#if 1
      A(i, j) = 2.0 * (float)drand48() - 1.0;
#else
      A(i, j) = (j - i) % 3;
#endif
    }
  }
}

float compare_matrices(int m, int n, float *a, float *b) {
  int i, j;
  float max_diff = 0.0, diff;
  int printed = 0;

  for (i = 0; i < m; ++i) {
    for (j = 0; j < n; ++j) {
      max_diff = abs(A(i, j) - B(i, j));
      if (printed == 0) {
        if (max_diff > 0.5f || max_diff < -0.5f) {
          printf("\n error: i %d j %d diff %f  got %f expect %f \n", i, j,
                 max_diff, A(i, j), B(i, j));
          printed = 1;
        }
      }
    }
  }
  return max_diff;
}

void cpu_sgemm(float *A_ptr, float *B_ptr, float *C_ptr, const int M,
               const int N, const int K) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = 0.f;
      for (int k = 0; k < K; ++k) {
        sum += A_ptr[m * K + k] * B_ptr[k * N + n];
      }
      C_ptr[m * N + n] = sum;
    }
  }
}

template <unsigned int BLOCK_SIZE, unsigned int STRIDE>
__global__ void cuda_sgemm(float *A_ptr, float *B_ptr, float *C_ptr,
                           const int M, const int N, const int K) {

  float *A_ptr_start = A_ptr + STRIDE * blockIdx.x * blockDim.x * K;
  float *B_ptr_start = B_ptr + STRIDE * blockIdx.y * blockDim.y;

  __shared__ float a_shared[BLOCK_SIZE * STRIDE][BLOCK_SIZE * STRIDE];
  __shared__ float b_shared[BLOCK_SIZE * STRIDE][BLOCK_SIZE * STRIDE];

  float temp[STRIDE][STRIDE] = {0.f};

  for (int s = 0; s < K; s += blockDim.x * STRIDE) {

    for (int i = 0; i < STRIDE; i++) {
      for (int j = 0; j < STRIDE; j++) {
        a_shared[threadIdx.x + i * BLOCK_SIZE][threadIdx.y + j * BLOCK_SIZE] =
            A_ptr_start[(threadIdx.x + i * BLOCK_SIZE) * K + threadIdx.y + s +
                        j * BLOCK_SIZE];
        b_shared[threadIdx.x + i * BLOCK_SIZE][threadIdx.y + j * BLOCK_SIZE] =
            B_ptr_start[(threadIdx.x + i * BLOCK_SIZE + s) * N + threadIdx.y +
                        j * BLOCK_SIZE];
      }
    }
    __syncthreads();

    for (int i = 0; i < STRIDE; i++) {
      for (int j = 0; j < STRIDE; j++) {
        for (int k = 0; k < BLOCK_SIZE * STRIDE; ++k) {
          temp[i][j] +=
              a_shared[threadIdx.x + i * BLOCK_SIZE][k] *
              b_shared[k][threadIdx.y + j * BLOCK_SIZE];
        }
      }
    }
    __syncthreads();
  }

  float *C_ptr_start = C_ptr + (blockIdx.x * blockDim.x * STRIDE) * N +
                       blockIdx.y * blockDim.y * STRIDE;
  for (int i = 0; i < STRIDE; i++) {
    for (int j = 0; j < STRIDE; j++) {
      C_ptr_start[(threadIdx.x + i * BLOCK_SIZE) * N + threadIdx.y +
                  j * BLOCK_SIZE] = temp[i][j];
    }
  }
}

int main() {
  int m = 1024;
  int n = 1024;
  int k = 1024;

  const size_t mem_size_A = m * k * sizeof(float);
  const size_t mem_size_B = k * n * sizeof(float);
  const size_t mem_size_C = m * n * sizeof(float);

  float *matrix_A_host = (float *)malloc(mem_size_A);
  float *matrix_B_host = (float *)malloc(mem_size_B);

  float *matrix_C_host_gpu_calc = (float *)malloc(mem_size_C);
  float *matrix_C_host_cpu_calc = (float *)malloc(mem_size_C);

  random_matrix(m, k, matrix_A_host);
  random_matrix(k, n, matrix_B_host);
  memset(matrix_C_host_gpu_calc, 0, mem_size_C);
  memset(matrix_C_host_cpu_calc, 0, mem_size_C);

  float *matrix_A_device, *matrix_B_device, *matrix_C_device;

  cudaMalloc((void **)&matrix_A_device, mem_size_A);
  cudaMalloc((void **)&matrix_B_device, mem_size_B);
  cudaMalloc((void **)&matrix_C_device, mem_size_C);

  cudaMemcpy(matrix_A_device, matrix_A_host, mem_size_A,
             cudaMemcpyHostToDevice);
  cudaMemcpy(matrix_B_device, matrix_B_host, mem_size_B,
             cudaMemcpyHostToDevice);

  cpu_sgemm(matrix_A_host, matrix_B_host, matrix_C_host_cpu_calc, m, n, k);

  constexpr int BLOCK = 16;
  constexpr int STRIDE = 4;
  dim3 block(BLOCK, BLOCK);
  dim3 grid((m + BLOCK - 1) / BLOCK / STRIDE, (n + BLOCK - 1) / BLOCK / STRIDE);
  cuda_sgemm<BLOCK, STRIDE><<<grid, block>>>(matrix_A_device, matrix_B_device,
                                             matrix_C_device, m, n, k);

  cudaMemcpy(matrix_C_host_gpu_calc, matrix_C_device, mem_size_C,
             cudaMemcpyDeviceToHost);

  float diff =
      compare_matrices(m, n, matrix_C_host_cpu_calc, matrix_C_host_gpu_calc);

  if (diff > 0.5f || diff < -0.5f) {
    printf("Error.\n");
    exit(-1);
  } else {
    printf("Success\n");
  }

  free(matrix_A_host);
  free(matrix_B_host);
  free(matrix_C_host_gpu_calc);
  free(matrix_C_host_cpu_calc);

  cudaFree(matrix_A_device);
  cudaFree(matrix_B_device);
  cudaFree(matrix_C_device);

  return 0;
}