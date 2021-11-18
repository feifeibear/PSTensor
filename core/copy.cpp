// Copyright (C) 2021 Jiarui Fang (fangjiarui123@gmail.com).
// All rights reserved.

#include <cuda_runtime.h>
#include "copy.h"
#include "cuda.h"

namespace ps_tensor {
namespace core {

void Copy(void *dst, const void* src, size_t size,
                        DLDeviceType dstDevice,
                        DLDeviceType srcDevice) {
    if (dstDevice == kDLGPU && srcDevice == kDLCPU) {
        cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    } else if (dstDevice == kDLCPU && srcDevice == kDLGPU) {
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    } else if (dstDevice == kDLCPU && srcDevice == kDLCPU) {
        cudaMemcpy(dst, src, size, cudaMemcpyHostToHost);
    } else if (dstDevice == kDLGPU && srcDevice == kDLGPU) {
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    }
}


}  // namespace core
}  // namespace ps_tensor
