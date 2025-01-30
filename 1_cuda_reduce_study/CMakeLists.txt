add_executable(my_reduce_v8_shuffle my_reduce_v8_shuffle.cu)

target_link_libraries(my_reduce_v8_shuffle PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_reduce_v8_shuffle PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()