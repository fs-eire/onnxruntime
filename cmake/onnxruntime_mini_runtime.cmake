# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# mini_runtime - experimental lightweight executable that links against the
# core ONNX Runtime static libraries (session, framework, graph, optimizer,
# providers, etc.) and their transitive external dependencies so that the
# executable can exercise runtime functionality directly.
#
# This target has no dependency on unit-test infrastructure.

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
set(MINI_RUNTIME_SRC_DIR ${ONNXRUNTIME_ROOT}/tool/mini_runtime)

file(GLOB mini_runtime_srcs CONFIGURE_DEPENDS
  "${MINI_RUNTIME_SRC_DIR}/*.cc"
  "${MINI_RUNTIME_SRC_DIR}/*.h"
)

# ---------------------------------------------------------------------------
# Statically-linked provider libs (same set that the test targets use,
# but computed here so we have no dependency on onnxruntime_unittests.cmake).
# Providers like CUDA, TensorRT, MIGRAPHX, DNNL, and OpenVINO are loaded
# dynamically at runtime, so they are NOT listed here.
# ---------------------------------------------------------------------------
set(mini_runtime_static_providers
    ${PROVIDERS_NNAPI}
    ${PROVIDERS_VSINPU}
    ${PROVIDERS_JS}
    ${PROVIDERS_SNPE}
    ${PROVIDERS_RKNPU}
    ${PROVIDERS_DML}
    ${PROVIDERS_ACL}
    ${PROVIDERS_ARMNN}
    ${PROVIDERS_COREML}
    ${PROVIDERS_XNNPACK}
    ${PROVIDERS_AZURE}
)

if (onnxruntime_BUILD_QNN_EP_STATIC_LIB)
  list(APPEND mini_runtime_static_providers onnxruntime_providers_qnn)
endif()
if (onnxruntime_USE_WEBGPU AND NOT onnxruntime_USE_EP_API_ADAPTERS)
  list(APPEND mini_runtime_static_providers onnxruntime_providers_webgpu)
endif()

# ---------------------------------------------------------------------------
# Internal libraries – core ONNX Runtime modules.
# ---------------------------------------------------------------------------
set(mini_runtime_libs
    onnxruntime_session
    ${onnxruntime_libs}                       # may be empty in non-training builds
    ${mini_runtime_static_providers}
    onnxruntime_optimizer
    onnxruntime_providers
    onnxruntime_util
    onnxruntime_lora
    onnxruntime_framework
    onnxruntime_util
    onnxruntime_graph
    ${ONNXRUNTIME_MLAS_LIBS}
    onnxruntime_common
    onnxruntime_flatbuffers
)

# ---------------------------------------------------------------------------
# Build-time dependencies – same as onnxruntime_test_providers_dependencies
# ---------------------------------------------------------------------------
set(mini_runtime_deps ${onnxruntime_EXTERNAL_DEPENDENCIES})

if(onnxruntime_USE_CUDA)
  list(APPEND mini_runtime_deps onnxruntime_providers_cuda)
endif()
if(onnxruntime_USE_CANN)
  list(APPEND mini_runtime_deps onnxruntime_providers_cann)
endif()
if(onnxruntime_USE_DML)
  list(APPEND mini_runtime_deps onnxruntime_providers_dml)
endif()
if(onnxruntime_USE_DNNL)
  list(APPEND mini_runtime_deps onnxruntime_providers_dnnl)
endif()
if(onnxruntime_USE_MIGRAPHX)
  list(APPEND mini_runtime_deps onnxruntime_providers_migraphx)
endif()
if(onnxruntime_USE_COREML)
  list(APPEND mini_runtime_deps onnxruntime_providers_coreml coreml_proto)
endif()
if(onnxruntime_USE_TENSORRT)
  list(APPEND mini_runtime_deps onnxruntime_providers_tensorrt onnxruntime_providers_shared)
  list(APPEND mini_runtime_libs ${TENSORRT_LIBRARY_INFER})
endif()
if(onnxruntime_USE_OPENVINO)
  list(APPEND mini_runtime_deps onnxruntime_providers_openvino onnxruntime_providers_shared)
endif()
if(onnxruntime_USE_NNAPI_BUILTIN)
  list(APPEND mini_runtime_deps onnxruntime_providers_nnapi)
endif()
if(onnxruntime_USE_VSINPU)
  list(APPEND mini_runtime_deps onnxruntime_providers_vsinpu)
endif()
if(onnxruntime_USE_JSEP)
  list(APPEND mini_runtime_deps onnxruntime_providers_js)
endif()
if(onnxruntime_USE_RKNPU)
  list(APPEND mini_runtime_deps onnxruntime_providers_rknpu)
endif()
if(onnxruntime_USE_ACL)
  list(APPEND mini_runtime_deps onnxruntime_providers_acl)
endif()
if(onnxruntime_USE_ARMNN)
  list(APPEND mini_runtime_deps onnxruntime_providers_armnn)
endif()
if(onnxruntime_USE_WEBGPU AND NOT onnxruntime_USE_EP_API_ADAPTERS)
  list(APPEND mini_runtime_deps onnxruntime_providers_webgpu)
endif()

# ---------------------------------------------------------------------------
# Executable target
# ---------------------------------------------------------------------------
onnxruntime_add_executable(mini_runtime ${mini_runtime_srcs})

set_target_properties(mini_runtime PROPERTIES FOLDER "ONNXRuntimeTool")

source_group(TREE ${REPO_ROOT} FILES ${mini_runtime_srcs})

if(mini_runtime_deps)
  list(REMOVE_DUPLICATES mini_runtime_deps)
  add_dependencies(mini_runtime ${mini_runtime_deps})
endif()

if(mini_runtime_libs)
  list(REMOVE_DUPLICATES mini_runtime_libs)
endif()

# Link internal libraries + external (system / third-party) libraries.
target_link_libraries(mini_runtime PRIVATE
  ${mini_runtime_libs}
  ${onnxruntime_EXTERNAL_LIBRARIES}
)

# CUDA-specific link dependencies (mirrors AddTest logic).
if(onnxruntime_USE_CUDA)
  target_link_libraries(mini_runtime PRIVATE CUDA::cudart)
  target_include_directories(mini_runtime PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  if(NOT onnxruntime_CUDA_MINIMAL)
    target_link_libraries(mini_runtime PRIVATE cudnn_frontend)
    target_include_directories(mini_runtime PRIVATE ${CUDNN_INCLUDE_DIR})
  endif()
endif()

# Include directories.
target_include_directories(mini_runtime PRIVATE
  ${ONNXRUNTIME_ROOT}                          # onnxruntime/
  ${REPO_ROOT}/include/onnxruntime             # public headers
)

onnxruntime_add_include_to_target(mini_runtime
  date::date
  flatbuffers::flatbuffers
  onnx
  onnx_proto
  Boost::mp11
  safeint_interface
)

if(MSVC)
  target_compile_options(mini_runtime PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
    "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>"
  )
  # Set VS debugger working directory next to the executable.
  set_target_properties(mini_runtime PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:mini_runtime>
  )
endif()

if(WIN32)
  target_link_libraries(mini_runtime PRIVATE debug dbghelp)
endif()
