add_executable(combined_access combined_access.cu)
target_link_libraries(combined_access PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(combined_access PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
target_compile_options(combined_access PRIVATE -lineinfo)

add_executable(my_transpose_v1_naive my_transpose_v1_naive.cu)
target_link_libraries(my_transpose_v1_naive PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_transpose_v1_naive PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
target_compile_options(my_transpose_v1_naive PRIVATE -lineinfo)