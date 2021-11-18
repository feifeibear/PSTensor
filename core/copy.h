// Copyright (C) 2021 Jiarui Fang (fangjiarui123@gmail.com).
// All rights reserved.

#pragma once
#include <dlpack/dlpack.h>
#include <vector>
namespace ps_tensor {
namespace core {
void Copy(void *dst, const void* src, size_t size,
                        DLDeviceType dstDevice,
                        DLDeviceType srcDevice);

}  // namespace ps_tensor
}  // namespace core
