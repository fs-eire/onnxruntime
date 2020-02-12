// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>

#ifdef __NVCC__
#include "core/providers/cuda/cu_inc/common.cuh"
#define ORT_DEVICE __device__
#define HelperMin(a, b) _Min(a, b)
#define HelperMax(a, b) _Max(a, b)
#else
#include <algorithm>
#define ORT_DEVICE
#define HelperMin(a, b) std::min(a, b)
#define HelperMax(a, b) std::max(a, b)
#endif

namespace onnxruntime {

template <typename T>
struct PrepareContext {
  const T* boxes_data_ = nullptr;
  int64_t boxes_size_ = 0ll;
  const T* scores_data_ = nullptr;
  int64_t scores_size_ = 0ll;
  // The below are ptrs since they cab be device specific
  const int64_t* max_output_boxes_per_class_ = nullptr;
  const T* score_threshold_ = nullptr;
  const T* iou_threshold_ = nullptr;
  int64_t num_batches_ = 0;
  int64_t num_classes_ = 0;
  int64_t num_boxes_ = 0;
};

struct SelectedIndex {
  ORT_DEVICE
  SelectedIndex(int64_t batch_index, int64_t class_index, int64_t box_index)
      : batch_index_(batch_index), class_index_(class_index), box_index_(box_index) {}
  SelectedIndex() = default;
  int64_t batch_index_ = 0;
  int64_t class_index_ = 0;
  int64_t box_index_ = 0;
};

#ifdef __NVCC__
namespace cuda {
#endif
namespace nms_helpers {

template <typename T>
ORT_DEVICE
inline void MaxMin(T lhs, T rhs, T& min, T& max) {
  if (lhs >= rhs) {
    min = rhs;
    max = lhs;
  } else {
    min = lhs;
    max = rhs;
  }
}

template <typename T>
ORT_DEVICE
inline bool SuppressByIOU(const T* boxes_data, int64_t box_index1, int64_t box_index2,
                          int64_t center_point_box, T iou_threshold) {
  T x1_min{};
  T y1_min{};
  T x1_max{};
  T y1_max{};
  T x2_min{};
  T y2_min{};
  T x2_max{};
  T y2_max{};

  const T* box1 = boxes_data + 4 * box_index1;
  const T* box2 = boxes_data + 4 * box_index2;
  // center_point_box_ only support 0 or 1
  if (0 == center_point_box) {
    // boxes data format [y1, x1, y2, x2],
    MaxMin(box1[1], box1[3], x1_min, x1_max);
    MaxMin(box1[0], box1[2], y1_min, y1_max);
    MaxMin(box2[1], box2[3], x2_min, x2_max);
    MaxMin(box2[0], box2[2], y2_min, y2_max);
  } else {
    // 1 == center_point_box_ => boxes data format [x_center, y_center, width, height]
    T box1_width_half = box1[2] / (T)2;
    T box1_height_half = box1[3] / (T)2;
    T box2_width_half = box2[2] / (T)2;
    T box2_height_half = box2[3] / (T)2;

    x1_min = box1[0] - box1_width_half;
    x1_max = box1[0] + box1_width_half;
    y1_min = box1[1] - box1_height_half;
    y1_max = box1[1] + box1_height_half;

    x2_min = box2[0] - box2_width_half;
    x2_max = box2[0] + box2_width_half;
    y2_min = box2[1] - box2_height_half;
    y2_max = box2[1] + box2_height_half;
  }

  const T intersection_x_min = HelperMax(x1_min, x2_min);
  const T intersection_y_min = HelperMax(y1_min, y2_min);
  const T intersection_x_max = HelperMin(x1_max, x2_max);
  const T intersection_y_max = HelperMin(y1_max, y2_max);

  const T intersection_area = HelperMax(intersection_x_max - intersection_x_min, (T).0f) *
                              HelperMax(intersection_y_max - intersection_y_min, (T).0f);

  if (intersection_area <= (T).0f) {
    return false;
  }

  const T area1 = (x1_max - x1_min) * (y1_max - y1_min);
  const T area2 = (x2_max - x2_min) * (y2_max - y2_min);
  const T union_area = area1 + area2 - intersection_area;

  if (area1 <= (T).0f || area2 <= (T).0f || union_area <= (T).0f) {
    return false;
  }

  const T intersection_over_union = intersection_area / union_area;

  return intersection_over_union > iou_threshold;
}
#ifdef __NVCC__
} // namespace cuda
#endif
}  // nms_helpers
}  // namespace onnxruntime
