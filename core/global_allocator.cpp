// Copyright (C) 2021 Jiarui Fang (fangjiarui123@gmail.com).
// All rights reserved.

// Copyright (C) 2021 Jiarui Fang (fangjiarui123@gmail.com).  All rights reserved.

#include "core/global_allocator.h"
#include "core/base_allocator.h"
#include "core/cache_allocator.h"
namespace ps_tensor {
namespace core {
namespace allocator {

struct Allocator::AllocatorImpl {
  AllocatorImpl() : allocator_ptr(new NaiveAllocator()) {}

  void* allocate(size_t size, DLDeviceType dev, const std::string& name) {
    return allocator_ptr->allocate(size, dev, name);
  }

  void free(void* memory, DLDeviceType dev, const std::string& name) {
    allocator_ptr->free(memory, dev, name);
  }
  std::unique_ptr<BaseAllocator> allocator_ptr;
};

Allocator::~Allocator() = default;
/**
 * Defualt constructor.
 * Dev : CPU or GPU
 * set a default schema. CPU uses naiveAllocator, GPU uses cubAllocator
 */
Allocator::Allocator() : impl_(new AllocatorImpl()) {
}

void* Allocator::allocate(size_t size, DLDeviceType dev,
                          const std::string& name) {
  return impl_->allocate(size, dev, name);
}

void Allocator::free(void* memory, DLDeviceType dev, const std::string& name) {
  impl_->free(memory, dev, name);
}

}  // namespace allocator
}  // namespace core
}  // namespace ps_tensor
