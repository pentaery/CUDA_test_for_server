cmake_minimum_required(VERSION 3.17.0)
project(cuda_best_practice VERSION 0.1.0 LANGUAGES CXX CUDA C)
find_package(CUDAToolkit)
# add_subdirectory(1_cuda_reduce_study)
# add_subdirectory(2_cuda_sgemm_study)
# add_subdirectory(3_kernal_profiling_guide)
add_subdirectory(4_test_sparse)