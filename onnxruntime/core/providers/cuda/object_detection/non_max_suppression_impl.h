// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/common/common.h"
#include "core/framework/allocator.h"
#include "core/framework/data_types.h"
#include <functional>
#include "core/providers/cpu/object_detection/non_max_suppression_helper.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
Status NonMaxSuppressionImpl(
    std::function<IAllocatorUniquePtr<void>(size_t)> allocator,
    const PrepareContext<T>& pc,
    const int64_t center_point_box,
    int64_t batch_index,
    int64_t class_index,
    int max_output_boxes_per_class,
    T iou_threshold,
    T score_threshold,
    IAllocatorUniquePtr<void>& selected_indices,
    int* h_number_selected);

}  // namespace cuda
}  // namespace onnxruntime
