// Copyright (C) 2021 Jiarui Fang (fangjiarui123@gmail.com).
// All rights reserved.

// Copyright (C) 2021 Jiarui Fang (fangjiarui123@gmail.com).  All rights reserved.

#include "tensor.h"
#include "core/global_allocator.h"

namespace ps_tensor {
namespace core {


static void DLManagedTensorDeletor(DLManagedTensor *self) {
  if (self == nullptr) {
    return;
  }
  if (self->dl_tensor.data != nullptr) {
    if (self->dl_tensor.device.device_type == kDLCPU ||
        self->dl_tensor.device.device_type == kDLCUDA) {
      allocator::Allocator &allocator = allocator::Allocator::GetInstance();
      allocator.free(self->dl_tensor.data, self->dl_tensor.device.device_type, "");
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

  newTensor->dl_tensor.device = {device, device_id};  // device_type, device_id
  newTensor->dl_tensor.ndim = shape_list.size();

  newTensor->dl_tensor.dtype = {
      static_cast<uint8_t>(data_type_code), static_cast<uint8_t>(bits),
      static_cast<uint16_t>(lanes)};  // code, bits, lanes

  newTensor->dl_tensor.strides = nullptr;  // TODO
  newTensor->dl_tensor.byte_offset = 0;

  size_t numel = std::accumulate(shape_list.begin(), shape_list.end(), 1,
                                 std::multiplies<int64_t>());
  if (device == kDLCPU || device == kDLCUDA) {
    size_t size = numel * (bits / 8);
    allocator::Allocator &allocator = allocator::Allocator::GetInstance();
    newTensor->dl_tensor.data = allocator.allocate(size, device);
    newTensor->deleter = DLManagedTensorDeletor;
  } else {
    // TT_THROW("only cpu and gpu are supported!");
  }

  return newTensor;
}

}  // namespace core
}  // namespace ps_tensor
