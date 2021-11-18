// Copyright (C) 2021 Jiarui Fang (fangjiarui123@gmail.com).
// All rights reserved.

#pragma once
#include <dlpack/dlpack.h>

namespace ps_tensor {
namespace core {
namespace common {

static inline size_t GetDataSize(const DLTensor* t) {
   size_t size = 1;
   for (auto i = 0; i < t->ndim; ++i) {
     size *= t->shape[i];
   }
   size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;
  return size;
}

} // namespace ps_tensor
} // namespace core
} // namespace common
