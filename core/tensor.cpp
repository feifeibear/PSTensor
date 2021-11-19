// Copyright (C) 2021 Jiarui Fang (fangjiarui123@gmail.com).
// All rights reserved.

// Copyright (C) 2021 Jiarui Fang (fangjiarui123@gmail.com).  All rights reserved.

#include "tensor.h"
#include "core/global_allocator.h"
#include "common.h"
#include "copy.h"

namespace ps_tensor {
namespace core {


static void DLManagedTensorDeletor(DLManagedTensor *self) {
  if (self == nullptr) {
    return;
  }
  if (self->dl_tensor.data != nullptr) {
    if (self->dl_tensor.ctx.device_type == kDLCPU ||
        self->dl_tensor.ctx.device_type == kDLGPU) {
      allocator::Allocator &allocator = allocator::Allocator::GetInstance();
      allocator.free(self->dl_tensor.data, self->dl_tensor.ctx.device_type, "");
    }
  }

  delete[] self->dl_tensor.shape;
  delete self;
}


DLManagedTensor *NewDLPackTensor(const std::vector<int64_t> &shape_list,
                                 DLDeviceType device, int device_id,
                                 uint8_t data_type_code, size_t bits,
                                 size_t lanes, const std::string &name) {
  TT_ENFORCE_NE(shape_list.size(), 0, "Shape list should not be empty");
  auto *newTensor = new DLManagedTensor();

  newTensor->dl_tensor.shape = new int64_t[shape_list.size()];
  std::copy(shape_list.begin(), shape_list.end(), newTensor->dl_tensor.shape);

  newTensor->dl_tensor.ctx = {device, device_id};  // device_type, device_id
  newTensor->dl_tensor.ndim = shape_list.size();

  newTensor->dl_tensor.dtype = {
      static_cast<uint8_t>(data_type_code), static_cast<uint8_t>(bits),
      static_cast<uint16_t>(lanes)};  // code, bits, lanes

  newTensor->dl_tensor.strides = nullptr;  // TODO
  newTensor->dl_tensor.byte_offset = 0;

  size_t numel = std::accumulate(shape_list.begin(), shape_list.end(), 1,
                                 std::multiplies<int64_t>());
  if (device == kDLCPU || device == kDLGPU) {
    size_t size = numel * (bits / 8);
    allocator::Allocator &allocator = allocator::Allocator::GetInstance();
    newTensor->dl_tensor.data = allocator.allocate(size, device);
    newTensor->deleter = DLManagedTensorDeletor;
  } else {
    TT_THROW("only cpu and gpu are supported!");
  }

  return newTensor;
}

Tensor Tensor::move(const std::string device_type) {
  auto &dl_tensor = to_dl_tensor();
  allocator::Allocator &allocator = allocator::Allocator::GetInstance();
  auto size = common::GetDataSize(&dl_tensor);
  DLDeviceType target_dev;
  if (device_type == "cpu") {
    target_dev = kDLCPU;
  } else if (device_type == "cuda") {
    target_dev = kDLGPU;
  }
  // std::cerr << "allocate " << size << " bytes on " << target_dev << std::endl;
  // void* dst = allocator.allocate(size, target_dev);
  // core::Copy(dst, dl_tensor.data, size, target_dev, dl_tensor.ctx.device_type);
  // dl_tensor.data = dst;
  // dl_tensor.ctx.device_type = target_dev;
  std::vector<int64_t> shape_vec;
  for (auto i = 0; i < dl_tensor.ndim; ++i) {
     shape_vec.push_back(dl_tensor.shape[i]);
   }
  auto t = NewDLPackTensor(shape_vec, target_dev, dl_tensor.ctx.device_id,
                         dl_tensor.dtype.code,
                         dl_tensor.dtype.bits,
                         dl_tensor.dtype.lanes, "");
  core::Copy(t->dl_tensor.data, dl_tensor.data, size, target_dev, dl_tensor.ctx.device_type);
  return std::move(Tensor(t));
}

Tensor Tensor::clone() {
  auto &dl_tensor = to_dl_tensor();
  auto size = common::GetDataSize(&dl_tensor);
  std::vector<int64_t> shape_vec;
  for (auto i = 0; i < dl_tensor.ndim; ++i) {
     shape_vec.push_back(dl_tensor.shape[i]);
   }
  auto t = NewDLPackTensor(shape_vec, dl_tensor.ctx.device_type, dl_tensor.ctx.device_id,
                         dl_tensor.dtype.code,
                         dl_tensor.dtype.bits,
                         dl_tensor.dtype.lanes, "");
  core::Copy(t->dl_tensor.data, dl_tensor.data, size, dl_tensor.ctx.device_type, dl_tensor.ctx.device_type);
  return std::move(Tensor(t));
}
}  // namespace core
}  // namespace ps_tensor
