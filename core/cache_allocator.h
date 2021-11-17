// Copyright (C) 2021 Jiarui Fang (fangjiarui123@gmail.com).
// All rights reserved.

// Copyright (C) 2021 Jiarui Fang (fangjiarui123@gmail.com).  All rights reserved.

#pragma once
#include "core/base_allocator.h"
#include <sstream>
#include <cuda_runtime.h>
#include "core/enforce.h"
#include <cub/util_allocator.cuh>
namespace ps_tensor {
namespace core {
namespace allocator {

struct BadAlloc : public std::exception {
  explicit BadAlloc(std::string err_msg) : err_str_(err_msg) {}

  const char *what() const noexcept override { return err_str_.c_str(); }

  std::string err_str_;
};

class NaiveAllocator : public BaseAllocator {
 public:
  NaiveAllocator() : cub_allocator(unsigned(8)) {}
  void* allocate(size_t size, DLDeviceType dev,
                 const std::string& name) override {
    void* data = nullptr;
    if (dev == kDLCPU) {
      return malloc(size);
    } else if (dev == kDLCUDA) {
      try {
        cudaError_t result = cub_allocator.DeviceAllocate(&data, size);
        if (result != cudaSuccess) {
          throw BadAlloc("DeviceAllocate failed.");
        }
      } catch (...) {
        cub_allocator.FreeAllCached();
        cudaError_t result = cub_allocator.DeviceAllocate(&data, size);
        if (result != cudaSuccess) {
          std::stringstream ss;
          ss << "DeviceAllocate failed Again. " << size;
          throw BadAlloc(ss.str());
        }
      }
    }
    return data;
  }  // alloc

  void free(void* mem, DLDeviceType dev, const std::string& name) override {
    if (dev == kDLCPU) {
      // TODO(jiaruifang) We can not delete an void*
      delete mem;
    } else if (dev == kDLCUDA) {
      try {
        cudaError_t result = cub_allocator.DeviceFree(mem);
        if (result != cudaErrorCudartUnloading && result != cudaSuccess) {
          TT_THROW("DeviceFree failed ");
        }
      } catch (...) {
      }
    }
  }

  ~NaiveAllocator();

 private:
  cub::CachingDeviceAllocator cub_allocator;
  /*
  (  unsigned int   bin_growth,
    unsigned int   min_bin = 1,
    unsigned int   max_bin = INVALID_BIN,
    size_t   max_cached_bytes = INVALID_SIZE,
    bool   skip_cleanup = false,
    bool   debug = false
  )
   */
};

}  // namespace allocator
}  // namespace core
}  // namespace ps_tensor
