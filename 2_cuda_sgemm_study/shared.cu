#include <cuda_runtime.h>
#include <iostream>

int main() {
  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);

  std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes"
            << std::endl;
  std::cout << "Shared Memory per SM: " << prop.sharedMemPerMultiprocessor
            << " bytes" << std::endl;
  return 0;
}