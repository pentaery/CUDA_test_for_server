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
      diff = abs(A(i, j) - B(i, j));
      if (printed == 0) {
        if (max_diff > 0.5f || max_diff < -0.5f) {
          printf("\n error: i %d j %d diff %f  got %f expect %f ", i, j,
                 max_diff, A(i, j), B(i, j));
          printed = 1;
        }
      }
    }
  }
  return max_diff;
}

void cpu_sgemm(float *A_ptr, float *B_ptr, float *C_ptr, int M, int N, int K) {
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

int main() {
  int m = 2048;
  int n = 2048;
  int k = 2048;

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
  dim3 block(BLOCK, BLOCK);
  dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

  cudaMemcpy(matrix_C_host_gpu_calc, matrix_C_device, mem_size_C,
             cudaMemcpyDeviceToHost);

  return 0;
}