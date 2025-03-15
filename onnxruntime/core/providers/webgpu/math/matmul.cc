// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/math/matmul.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

#include "core/providers/webgpu/data_transfer.h"
namespace onnxruntime {
namespace webgpu {


ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    MatMul,
    kOnnxDomain,
    1, 12,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    MatMul);

ONNX_OPERATOR_KERNEL_EX(
    MatMul,
    kOnnxDomain,
    13,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    MatMul);

// function that prints GPU tensor
Status MatMul::PrintGPUTensor(ComputeContext& context, const Tensor& tensor) const {
  // first create Temp tensor on CPU
  Tensor temp_tensor = context.CreateCPUTensor(tensor.DataType(), tensor.Shape());
  // copy data from GPU to CPU
  ORT_RETURN_IF_ERROR(Info().GetDataTransferManager().CopyTensor(tensor, temp_tensor));

  // print data
  auto data = temp_tensor.Data<float>();

  for (int i = 0; i < temp_tensor.Shape().Size(); i++) {
    LOGS_DEFAULT(VERBOSE) << data[i] << " ";
  }
  LOGS_DEFAULT(VERBOSE) << "\n";

  return Status::OK();
}

std::string CalcResult(int components, int a_components, int output_number) {
    std::ostringstream oss;
    oss << "var a_data: a_value_t;\n";
    for (int i = 0; i < a_components; ++i) {
        oss << "let b_data" << i << " = b[(b_offset + (k + " << i << ") * uniforms.N + col) / " << components << "];\n";
    }
    for (int i = 0; i < output_number; ++i) {
        oss << "a_data = a[(a_offset + (row + " << i << ") * uniforms.K + k) / " << a_components << "];\n";

        for (int j = 0; j < a_components; j++) {
            oss << "values[" << i << "] = fma(b_value_t(a_data" << (a_components == 1 ? "" : "[" + std::to_string(j) + "]") << "), b_data" << j << ", values[" << i << "]);\n";
        }
    }
    return oss.str();

}
// matrix multiplication: MxK * KxN = MxN
Status MatMulNativeProgram::GenerateShaderCode(ShaderHelper& shader) const {

    LOGS_DEFAULT(VERBOSE) << "MatMulNativeProgram: Start generating shader code";
    const auto& a = shader.AddInput("a", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias| ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

    std::string process_bias;
    if (has_bias_) {
        shader.AddInput("bias", ShaderUsage::UseUniform);
        process_bias = "value += output_value_t(bias[row +i]);";
    }

    const auto& output = shader.AddOutput("output",ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
    const auto& batch_dims = shader.AddIndices("batch_dims");

    int a_components = a.NumComponents();
    int components = b.NumComponents(); // components of N

    shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                              << "let col = (global_idx % (uniforms.N / " << components << ")) * " << components << ";\n"
                              << "var index1 = global_idx / (uniforms.N / " << components << ");\n"
                              << "let stride1 = uniforms.M / " << output_number_ << ";\n"
                              << "let row = (index1 % stride1) * " << output_number_ << ";\n"
                              << "let batch = index1 / stride1;\n";
    if (output_size_ != 2) {
        shader.MainFunctionBody() << "let batch_indices = " << batch_dims.OffsetToIndices("batch") << ";\n";
    }
    shader.MainFunctionBody() << "var a_indices: a_indices_t;\n"
                              << ConvertOutputBatchIndicesToInputBatchIndices("a", a, a.Rank() - 2, batch_dims.Rank(), "batch_indices")
                              << a.IndicesSet("a_indices", a.Rank() - 2, 0) << "\n"
                              << a.IndicesSet("a_indices", a.Rank() - 1, 0) << "\n"
                              << "let a_offset = " << a.IndicesToOffset("a_indices") << " * "<<a_components<<";\n"
                              << "var b_indices: b_indices_t;\n"
                              << ConvertOutputBatchIndicesToInputBatchIndices("b", b, b.Rank() - 2, batch_dims.Rank(), "batch_indices")
                              << b.IndicesSet("b_indices", b.Rank() - 2, 0) << "\n"
                              << b.IndicesSet("b_indices", b.Rank() - 1, 0) << "\n"
                              << "let b_offset = " << b.IndicesToOffset("b_indices") << "*"<<components<<";\n"
                              << "var values: array<output_value_t, " << output_number_ << ">;\n"
                              << "for (var k: u32 = 0u; k < uniforms.K; k = k + " << a_components << ") {\n"
                              << CalcResult(components, a_components, output_number_) << "\n"
                              << "}\n"
                              << "for (var i = 0u; i < " << output_number_ << "u; i++) {\n"
                              << "  var value = values[i];\n"
                              << process_bias << "\n"
                              << "  let cur_indices = output_indices_t(batch, row + i, col/"<<components<<");\n"
                              << "  let offset = " << output.IndicesToOffset("cur_indices") << ";\n"
                              << output.SetByOffset("offset", "value")
                              << "}\n";


    return Status::OK();
}

Status MatMul::ComputeInternal(ComputeContext& context) const {
    LOGS_DEFAULT(VERBOSE) << "Running MatMul WebGPU kernel";

    // calculate output shape
    MatMulComputeHelper helper;
    const auto* a = context.Input(0);
    const auto* b = context.Input(1);

    ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
    auto* output_tensor = context.Output(0, helper.OutputShape());

    const uint32_t m = static_cast<uint32_t>(helper.M());
    const uint32_t n = static_cast<uint32_t>(helper.N());
    const uint32_t k = static_cast<uint32_t>(helper.K());


    LOGS_DEFAULT(VERBOSE) << "MatMulProgram: m: " << m;
    LOGS_DEFAULT(VERBOSE) << "MatMulProgram: n: " << n;
    LOGS_DEFAULT(VERBOSE) << "MatMulProgram: k: " << k;


    bool has_bias = context.InputCount() > 2;
    LOGS_DEFAULT(VERBOSE) << "MatMulProgram: has_bias: " << has_bias;

    if (n < 8 && k < 8) {  // call MatMulNativeProgram

        LOGS_DEFAULT(VERBOSE) << "Running MatMulNativeProgram";
        const int components = GetMaxComponents(n);
        const int a_components = GetMaxComponents(k);

        const int output_number = GetMaxComponents(m);
        uint32_t output_size = static_cast<uint32_t>(helper.OutputShape().Size() / components / output_number);

        const size_t output_rank = helper.OutputShape().NumDimensions();
        TensorShape outer_dims = output_rank > 2 ? helper.OutputShape().Slice(0, output_rank -2) : TensorShape({});
        const int64_t batch_size = outer_dims.Size();
        TensorShape output_shape_shader({batch_size, helper.M(), helper.N() / components});

        // logs
        LOGS_DEFAULT(VERBOSE) << "MatMulNativeProgram: a_shape: " << a->Shape().ToString();
        LOGS_DEFAULT(VERBOSE) << "MatMulNativeProgram: b_shape: " << b->Shape().ToString();
        LOGS_DEFAULT(VERBOSE) << "MatMulNativeProgram: outputshape: " << helper.OutputShape().ToString();

        LOGS_DEFAULT(VERBOSE) << "MatMulNativeProgram: components: " << components;
        LOGS_DEFAULT(VERBOSE) << "MatMulNativeProgram: a_components: " << a_components;
        LOGS_DEFAULT(VERBOSE) << "MatMulNativeProgram: output_size: " << output_size;
        LOGS_DEFAULT(VERBOSE) << "MatMulNativeProgram: output_number: " << output_number;
        LOGS_DEFAULT(VERBOSE) << "MatMulNativeProgram: output_rank: " << output_rank;
        LOGS_DEFAULT(VERBOSE) << "MatMulNativeProgram: outer_dims: " <<  outer_dims;
        LOGS_DEFAULT(VERBOSE) << "MatMulNativeProgram: batch_size: " << batch_size;
        LOGS_DEFAULT(VERBOSE) << "MatMulNativeProgram: output_shape_shader: " << output_shape_shader.ToString();

        MatMulNativeProgram program{output_size, output_number, has_bias};
        program
            .CacheHint(std::to_string(components), std::to_string(a_components), std::to_string(output_number))
            .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, a_components},
                        {b, ProgramTensorMetadataDependency::TypeAndRank, components}});

        if (has_bias) {
            const auto* bias = context.Input(2);
            program.AddInput({bias, ProgramTensorMetadataDependency::Rank, 1});
        }
        program
            .AddOutputs({{output_tensor, ProgramTensorMetadataDependency::None ,output_shape_shader, components}})
            .SetDispatchGroupSize((uint32_t)ceil(static_cast<float>(output_size) / 64 ))
            .AddIndices(outer_dims)
            .AddUniformVariables({{output_size}, {m}, {n}, {k}});

           // .AddUniformVariable({output_size});


        Status res = context.RunProgram(program);
        // reshape output shape to original shape
        output_tensor->Reshape(helper.OutputShape());
        LOGS_DEFAULT(VERBOSE) << "Output: ";
        ORT_RETURN_IF_ERROR(PrintGPUTensor(context, *output_tensor));
        return res;
    }

    LOGS_DEFAULT(VERBOSE) << "Running MatMulPackedProgram";

    int64_t batchA = a->Shape().SizeToDimension(a->Shape().NumDimensions() - 2);
    int64_t batchB = b->Shape().SizeToDimension(b->Shape().NumDimensions() - 2);

    TensorShape a_shape = a->Shape();
    TensorShape b_shape = b->Shape();
    TensorShape output_shape = helper.OutputShape();

    LOGS_DEFAULT(VERBOSE) << "MatMulPackedProgram: a_shape: " << a_shape.ToString();
    LOGS_DEFAULT(VERBOSE) << "MatMulPackedProgram: b_shape: " << b_shape.ToString();
    LOGS_DEFAULT(VERBOSE) << "MatMulPackedProgram: outputshape: " << output_shape.ToString();
    LOGS_DEFAULT(VERBOSE) << "MatMulPackedProgram: batchA: " << batchA;
    LOGS_DEFAULT(VERBOSE) << "MatMulPackedProgram: batchB: " << batchB;



    const int64_t m_value = output_shape[output_shape.NumDimensions() - 2];
    // check if A is  batch of vector (bach is not 1, M is 1) and B is a matrix (batch is 1)
    if (batchA != 1 && m_value == 1 && batchB == 1) {

        // optimization for batched vector matrix multiplication
        // dimensions of A: [1,`batchA`,K]
        TensorShapeVector dims_a = {1, batchA, helper.K()};
        // dimensions of B: [1,K,N]
        TensorShapeVector dims_b = {1, helper.K(), helper.N()};

        a_shape = TensorShape(dims_a);
        b_shape = TensorShape(dims_b);
        output_shape = {1, batchA, helper.N()};

        LOGS_DEFAULT(VERBOSE) << "MatMulPackedProgram: Apply optimization for batched vector matrix multiplication";
        LOGS_DEFAULT(VERBOSE) << "MatMulPackedProgram: new a_shape: " << a_shape.ToString();
        LOGS_DEFAULT(VERBOSE) << "MatMulPackedProgram: new b_shape: " << b_shape.ToString();
        LOGS_DEFAULT(VERBOSE) << "MatMulPackedProgram: new outputshape: " << output_shape.ToString();
    }

    // helpful dimension variables
    TensorShape outer_dims_a = a_shape.NumDimensions() > 2
                                ? a_shape.Slice(0, a_shape.NumDimensions() - 2)
                                : TensorShape({});

    TensorShape outer_dims_b = b_shape.NumDimensions() > 2
                                ? b_shape.Slice(0, b_shape.NumDimensions() - 2)
                                : TensorShape({});

    TensorShape outer_dims = output_shape.NumDimensions() > 2
                            ? output_shape.Slice(0, output_shape.NumDimensions() -2)
                            :TensorShape({});

    const int64_t batch_size = outer_dims.Size();

    LOGS_DEFAULT(VERBOSE) << "MatMulPackedProgram: outer_dims_a: " << outer_dims_a.ToString();
    LOGS_DEFAULT(VERBOSE) << "MatMulPackedProgram: outer_dims_b: " << outer_dims_b.ToString();
    LOGS_DEFAULT(VERBOSE) << "MatMulPackedProgram: outer_dims: " << outer_dims.ToString();
    LOGS_DEFAULT(VERBOSE) << "MatMulPackedProgram: batch_size: " << batch_size;


    // Get dimensions for matrix multiplication from TensorShape
    const int32_t dim_a_outer = static_cast<int32_t>(a_shape[a_shape.NumDimensions() - 2]);  // M dimension
    const int32_t dim_inner = static_cast<int32_t>(a_shape[a_shape.NumDimensions() - 1]);    // K dimension
    const int32_t dim_b_outer = static_cast<int32_t>(b_shape[b_shape.NumDimensions() - 1]);  // N dimension

    const bool is_vec4 = dim_inner % 4 == 0 && dim_b_outer % 4 == 0;


    InlinedVector<int64_t> elements_per_thread = dim_a_outer <= 8
                                                 ? InlinedVector<int64_t>({4, 1, 1})
                                                 : InlinedVector<int64_t>({4, 4, 1});

    const uint32_t dispatch_x = (uint32_t)ceil(static_cast<float>(dim_b_outer) / MATMUL_PACKED_WORKGROUP_SIZE_X / elements_per_thread[0]);
    const uint32_t dispatch_y = (uint32_t)ceil(static_cast<float>(dim_a_outer) / MATMUL_PACKED_WORKGROUP_SIZE_Y / elements_per_thread[1]);
    const uint32_t dispatch_z = (uint32_t)ceil(static_cast<float>(batch_size) / MATMUL_PACKED_WORKGROUP_SIZE_Z / elements_per_thread[2]);

    const int components = is_vec4 ? 4 : 1;
    const TensorShape a_shape_temp = BuildTempShapeVector(outer_dims_a, dim_a_outer, dim_inner, components);
    const TensorShape b_shape_temp = BuildTempShapeVector(outer_dims_b, dim_inner, dim_b_outer, components);
    const TensorShape output_shape_temp = TensorShape({batch_size , dim_a_outer, dim_b_outer / components});

    LOGS_DEFAULT(VERBOSE) << "MatMulPackedProgram: dim_a_outer: " << dim_a_outer;
    LOGS_DEFAULT(VERBOSE) << "MatMulPackedProgram: dim_inner: " << dim_inner;
    LOGS_DEFAULT(VERBOSE) << "MatMulPackedProgram: dim_b_outer: " << dim_b_outer;
    LOGS_DEFAULT(VERBOSE) << "MatMulPackedProgram: a_shape_temp: " << a_shape_temp.ToString();
    LOGS_DEFAULT(VERBOSE) << "MatMulPackedProgram: b_shape_temp: " << b_shape_temp.ToString();
    LOGS_DEFAULT(VERBOSE) << "MatMulPackedProgram: outputshape_temp: " << output_shape_temp.ToString();
    LOGS_DEFAULT(VERBOSE) << "MatMulPackedProgram: dispatch_x: " << dispatch_x;
    LOGS_DEFAULT(VERBOSE) << "MatMulPackedProgram: dispatch_y: " << dispatch_y;
    LOGS_DEFAULT(VERBOSE) << "MatMulPackedProgram: dispatch_z: " << dispatch_z;
    LOGS_DEFAULT(VERBOSE) << "MatMulPackedProgram: is_vec4: " << is_vec4;
    LOGS_DEFAULT(VERBOSE) << "MatMulPackedProgram: components: " << components;

    // elements per thread print
    LOGS_DEFAULT(VERBOSE) << "MatMulPackedProgram: elements_per_thread: ";
    for (const auto& elem : elements_per_thread) {
        LOGS_DEFAULT(VERBOSE) << elem << " ";
    }
    LOGS_DEFAULT(VERBOSE) << "\n";


    MatMulProgram program{has_bias, is_vec4, elements_per_thread};
    program
        .CacheHint(absl::StrJoin(elements_per_thread, "-"), std::to_string(is_vec4))
        .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, a_shape_temp, components},
                    {b, ProgramTensorMetadataDependency::TypeAndRank, b_shape_temp, components}})
        .AddOutputs({{output_tensor, ProgramTensorMetadataDependency::Rank, output_shape_temp, components}})
        .AddUniformVariables({{dim_a_outer}, {dim_b_outer}, {dim_inner}})
        .AddIndices(outer_dims)
        .SetDispatchGroupSize(dispatch_x, dispatch_y, dispatch_z)
        .SetWorkgroupSize(MATMUL_PACKED_WORKGROUP_SIZE_X, MATMUL_PACKED_WORKGROUP_SIZE_Y, MATMUL_PACKED_WORKGROUP_SIZE_Z);

    if (has_bias) {
        const auto* bias = context.Input(2);
        program.AddInput({bias, ProgramTensorMetadataDependency::Rank, 1});
    }


    Status res = context.RunProgram(program);
    // reshape output shape to original shape
    output_tensor->Reshape(helper.OutputShape());
    LOGS_DEFAULT(VERBOSE) << "Output: ";
    ORT_RETURN_IF_ERROR(PrintGPUTensor(context, *output_tensor));
    return res;
}


}  // namespace webgpu
}  // namespace onnxruntime
