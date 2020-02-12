// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "range.h"

#include <cmath>
#include "core/util/math.h"

namespace onnxruntime {

// Register a kernel for kMsDomain (contrib op) Range
#ifndef DISABLE_CONTRIB_OPS

namespace contrib {
// TODO: Remove this contrib kernel registration and the schema from the appropriate places
// once Keras Mask RCNN is shipped with all ONNX domain ops

// Currently this kernel is required to support Keras Mask-RCNN
ONNX_OPERATOR_KERNEL_EX(
    Range,  //name
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                            DataTypeImpl::GetTensorType<double>(),
                                            DataTypeImpl::GetTensorType<int16_t>(),
                                            DataTypeImpl::GetTensorType<int32_t>(),
                                            DataTypeImpl::GetTensorType<int64_t>()}),
    Range);

}  // namespace contrib

#endif

ONNX_CPU_OPERATOR_KERNEL(
    Range,
    11,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                            DataTypeImpl::GetTensorType<MLFloat16>(),
                                            DataTypeImpl::GetTensorType<double>(),
                                            DataTypeImpl::GetTensorType<int16_t>(),
                                            DataTypeImpl::GetTensorType<int32_t>(),
                                            DataTypeImpl::GetTensorType<int64_t>()}),
    Range);

namespace {
template <typename T>
Status Impl(OpKernelContext* ctx, T start, T limit, const T* p_delta);

template <typename T>
Status Impl(OpKernelContext* ctx, T start, T limit, const T* p_delta) {
  T delta = (p_delta == nullptr) ? T{1} : *p_delta;

  if (delta == T{0}) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "delta in Range operator can not be zero!");
  }
  int64_t n = static_cast<int64_t>(ceil((1.0 * (limit - start)) / delta));
  if (n <= 0)
    n = 0;
  TensorShape shape = {n};
  T* y = ctx->Output(0, shape)->template MutableData<T>();
  for (int64_t i = 0; i < n; ++i) {
    *y++ = start;
    start += delta;
  }

  return Status::OK();
}

template <>
Status Impl(OpKernelContext* ctx, MLFloat16 start, MLFloat16 limit, const MLFloat16* p_delta) {
  float f_start = math::halfToFloat(start.val);
  float f_limit = math::halfToFloat(limit.val);
  float f_delta = (p_delta == nullptr) ? 1.0 : math::halfToFloat(p_delta->val);

  if (f_delta == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "delta in Range operator can not be zero!");
  }
  int64_t n = static_cast<int64_t>(ceil((1.0 * (f_limit - f_start)) / f_delta));
  if (n <= 0)
    n = 0;
  TensorShape shape = {n};
  uint16_t* y = reinterpret_cast<uint16_t*>(ctx->Output(0, shape)->template MutableData<MLFloat16>());
  for (int64_t i = 0; i < n; ++i) {
    *y++ = math::floatToHalf(f_start);
    f_start += f_delta;
  }

  return Status::OK();
}

}

template <typename T>
static Status ComputeRange(OpKernelContext* ctx) {
  const auto& start_tensor = *ctx->Input<Tensor>(0);
  const auto& limit_tensor = *ctx->Input<Tensor>(1);
  const auto* delta_tensor_ptr = ctx->Input<Tensor>(2);

  if (!start_tensor.Shape().IsScalar()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "start in Range operator should be scalar like tensor, yet got shape:",
                           start_tensor.Shape());
  }
  if (!limit_tensor.Shape().IsScalar()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "limit in Range operator should be scalar like tensor, yet got shape:",
                           limit_tensor.Shape());
  }
  if (delta_tensor_ptr != nullptr && !delta_tensor_ptr->Shape().IsScalar()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "delta in Range operator should be scalar like tensor, yet got shape:",
                           delta_tensor_ptr->Shape());
  }

  // T start = *start_tensor.template Data<T>();
  // T limit = *limit_tensor.template Data<T>();
  // T delta = (delta_tensor_ptr == nullptr) ? T{1} : *(delta_tensor_ptr->template Data<T>());

  return Impl<T>(ctx, *start_tensor.template Data<T>(), *limit_tensor.template Data<T>(), delta_tensor_ptr == nullptr ? nullptr : delta_tensor_ptr->template Data<T>());
}

namespace range_internal {
template <class T>
struct CallRangeImpl {
  Status operator()(OpKernelContext* ctx) const {
    return ComputeRange<T>(ctx);
  }
};
}  // namespace range_internal

Status Range::Compute(OpKernelContext* ctx) const {
  const auto* input_tensor = ctx->Input<Tensor>(0);
  if (input_tensor == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  utils::MLTypeCallDispatcherRet<Status, range_internal::CallRangeImpl,
                                 int32_t, float, int64_t, double, int16_t, MLFloat16>
      t_disp(input_tensor->GetElementType());
  return t_disp.Invoke(ctx);
}

}  // namespace onnxruntime
