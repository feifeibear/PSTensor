#pragma once
#include <dlpack/dlpack.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

// Disable the copy and assignment operator for a class.
#ifndef DISABLE_COPY_AND_ASSIGN
#define DISABLE_COPY_AND_ASSIGN(classname)         \
 private:                                          \
  classname(const classname&) = delete;            \
  classname(classname&&) = delete;                 \
  classname& operator=(const classname&) = delete; \
  classname& operator=(classname&&) = delete
#endif

namespace ps_tensor {
namespace core {
namespace allocator {

/***
 * If the runtime detect the GPU, then init a GPU allocator as well as a CPU
 * one. If no GPU detected, only init a CPU allocator. In this way, we have to
 * pass a device parameter to the allocate and free API. The device type have to
 * be determined when call allocate.
 */
class Allocator {
 public:
  ~Allocator();

  static Allocator& GetInstance() {
    static Allocator instance;
    return instance;
  }

  void* allocate(size_t size, DLDeviceType dev, const std::string& name = "");
  void free(void* memory, DLDeviceType dev, const std::string& name = "");

 private:
  Allocator();
  struct AllocatorImpl;
  std::unique_ptr<AllocatorImpl> impl_;

  DISABLE_COPY_AND_ASSIGN(Allocator);
};
}  // namespace allocator
}  // namespace core
}  // namespace ps_tensor