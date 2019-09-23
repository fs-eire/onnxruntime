// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/framework/data_types.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace cuda {

void NonMaxSuppressionImpl();

}  // namespace cuda
}  // namespace onnxruntime
