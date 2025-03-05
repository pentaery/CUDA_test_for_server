#include <cassert>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      std::cerr << "CUDA Error: " << cudaGetErrorString(status) << " at "      \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      assert(false);                                                           \
    }                                                                          \
  }

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      std::cerr << "cuSPARSE Error: " << status << " at " << __FILE__ << ":"   \
                << __LINE__ << std::endl;                                      \
      assert(false);                                                           \
    }                                                                          \
  }

int main() {
  // 初始化 cuSPARSE
  cusparseHandle_t handle;
  CHECK_CUSPARSE(cusparseCreate(&handle));

  // 定义稀疏矩阵 (4x4) 和向量
  int m = 4, n = 4, nnz = 6;
  float h_csrValA[] = {1, 2, 3, 4, 5, 6};  // 非零元素值
  int h_csrRowPtrA[] = {0, 2, 3, 5, 6};    // 行指针
  int h_csrColIndA[] = {0, 3, 2, 0, 3, 1}; // 列索引
  float h_x[] = {1, 1, 1, 1};              // 输入向量
  float h_y[m] = {0};                      // 输出向量

  // 分配设备内存
  float *d_csrValA, *d_x, *d_y;
  int *d_csrRowPtrA, *d_csrColIndA;
  CHECK_CUDA(cudaMalloc((void **)&d_csrValA, nnz * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_csrRowPtrA, (m + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&d_csrColIndA, nnz * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&d_x, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_y, m * sizeof(float)));

  // 拷贝数据到设备
  CHECK_CUDA(cudaMemcpy(d_csrValA, h_csrValA, nnz * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA, (m + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_csrColIndA, h_csrColIndA, nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice));

  // 创建稀疏矩阵描述符
  cusparseSpMatDescr_t matA;
  CHECK_CUSPARSE(cusparseCreateCsr(&matA, m, n, nnz, d_csrRowPtrA, d_csrColIndA,
                                   d_csrValA, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_32F));

  // 创建向量描述符
  cusparseDnVecDescr_t vecX, vecY;
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_32F));
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, m, d_y, CUDA_R_32F));

  // 计算所需的外部缓冲区大小
  size_t bufferSize = 0;
  float alpha = 1.0f, beta = 0.0f;
  CHECK_CUSPARSE(cusparseSpMV_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY,
      CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

  // 分配外部缓冲区
  void *dBuffer = nullptr;
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

  // 执行稀疏矩阵-向量乘法 (SpMV)
  CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                              matA, vecX, &beta, vecY, CUDA_R_32F,
                              CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

  // 拷贝结果回主机
  CHECK_CUDA(cudaMemcpy(h_y, d_y, m * sizeof(float), cudaMemcpyDeviceToHost));

  // 打印结果
  std::cout << "Result y: ";
  for (int i = 0; i < m; i++) {
    std::cout << h_y[i] << " ";
  }
  std::cout << std::endl;

  // 释放资源
  CHECK_CUDA(cudaFree(d_csrValA));
  CHECK_CUDA(cudaFree(d_csrRowPtrA));
  CHECK_CUDA(cudaFree(d_csrColIndA));
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_y));
  CHECK_CUDA(cudaFree(dBuffer));
  CHECK_CUSPARSE(cusparseDestroySpMat(matA));
  CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
  CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
  CHECK_CUSPARSE(cusparseDestroy(handle));

  return 0;
}