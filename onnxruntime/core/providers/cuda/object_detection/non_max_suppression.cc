// Copyright (c) Microsoft Corporation. All rights reserved. 
// Licensed under the MIT License. 

#include "non_max_suppresion.h"
#include "non_max_suppresion_impl.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    NonMaxSuppression,
    kOnnxDomain,
    10,
    kCudaExecutionProvider,
    KernelDefBuilder(),
    NonMaxSuppression);

Status NonMaxSuppression::ComputeInternal(OpKernelContext* context) const {
  NonMaxSuppressionImpl(
      
  );

  return Status::OK();
}

}  // namespace cuda
};  // namespace onnxruntime
