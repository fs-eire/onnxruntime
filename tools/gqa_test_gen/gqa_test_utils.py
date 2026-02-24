"""
Shared utilities for generating GQA WebGPU test data.

Creates ONNX models matching the WebGPU GQA kernel's expected format (BNSH past/present),
runs them through ORT CPU, and formats the results as JSONC.
"""

import json
import math
import numpy as np
from onnx import TensorProto, helper
from onnxruntime import InferenceSession, SessionOptions


def ort_type_to_jsonc_type(ort_type):
    if ort_type == TensorProto.FLOAT:
        return "float32"
    elif ort_type == TensorProto.FLOAT16:
        return "float16"
    elif ort_type == TensorProto.INT32:
        return "int32"
    elif ort_type == TensorProto.INT64:
        return "int64"
    raise ValueError(f"Unsupported ORT type: {ort_type}")


def ort_type_to_np_dtype(ort_type):
    if ort_type == TensorProto.FLOAT:
        return np.float32
    elif ort_type == TensorProto.FLOAT16:
        return np.float16
    elif ort_type == TensorProto.INT32:
        return np.int32
    elif ort_type == TensorProto.INT64:
        return np.int64
    raise ValueError(f"Unsupported ORT type: {ort_type}")


def create_gqa_onnx_model(
    batch_size,
    seq_len,
    num_heads,
    kv_num_heads,
    head_size,
    past_seq_len=0,
    packed=False,
    rotary=False,
    rotary_interleaved=False,
    softcap=0.0,
    smooth_softmax=False,
    local_window_size=-1,
    has_position_ids=False,
    has_attention_bias=False,
    has_head_sink=False,
    ort_type=TensorProto.FLOAT,
):
    """Create an ONNX model for GroupQueryAttention matching WebGPU kernel format."""
    total_seq_len = past_seq_len + seq_len
    hidden_size = num_heads * head_size
    kv_hidden_size = kv_num_heads * head_size
    query_dim = hidden_size if not packed else (hidden_size + 2 * kv_hidden_size)

    # Node input names (empty string = absent optional input)
    node_inputs = [
        "query",
        "key" if not packed else "",
        "value" if not packed else "",
        "past_key",
        "past_value",
        "seqlens_k",
        "total_sequence_length",
        "cos_cache" if rotary else "",
        "sin_cache" if rotary else "",
        "position_ids" if has_position_ids else "",
        "attention_bias" if has_attention_bias else "",
        "head_sink" if has_head_sink else "",
    ]
    # Strip trailing empty strings
    while node_inputs and node_inputs[-1] == "":
        node_inputs.pop()

    node = helper.make_node(
        "GroupQueryAttention",
        node_inputs,
        ["output", "present_key", "present_value"],
        "GroupQueryAttention_0",
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
        local_window_size=local_window_size,
        do_rotary=1 if rotary else 0,
        rotary_interleaved=1 if rotary_interleaved else 0,
        softcap=softcap,
        smooth_softmax=1 if smooth_softmax else 0,
        domain="com.microsoft",
    )

    # Graph inputs
    graph_inputs = [
        helper.make_tensor_value_info("query", ort_type, [batch_size, seq_len, query_dim]),
    ]
    if not packed:
        graph_inputs += [
            helper.make_tensor_value_info("key", ort_type, [batch_size, seq_len, kv_hidden_size]),
            helper.make_tensor_value_info("value", ort_type, [batch_size, seq_len, kv_hidden_size]),
        ]
    graph_inputs += [
        helper.make_tensor_value_info("past_key", ort_type, [batch_size, kv_num_heads, past_seq_len, head_size]),
        helper.make_tensor_value_info("past_value", ort_type, [batch_size, kv_num_heads, past_seq_len, head_size]),
        helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [batch_size]),
        helper.make_tensor_value_info("total_sequence_length", TensorProto.INT32, [1]),
    ]
    if rotary:
        rotary_dim = (math.floor(head_size / 16) * 16) // 2
        graph_inputs += [
            helper.make_tensor_value_info("cos_cache", ort_type, [total_seq_len, rotary_dim]),
            helper.make_tensor_value_info("sin_cache", ort_type, [total_seq_len, rotary_dim]),
        ]
    if has_position_ids:
        graph_inputs += [
            helper.make_tensor_value_info("position_ids", TensorProto.INT64, [batch_size, seq_len]),
        ]
    if has_attention_bias:
        graph_inputs += [
            helper.make_tensor_value_info("attention_bias", ort_type, [batch_size, 1, seq_len, total_seq_len]),
        ]
    if has_head_sink:
        graph_inputs += [
            helper.make_tensor_value_info("head_sink", ort_type, [num_heads]),
        ]

    # Graph outputs (BNSH format for present KV)
    graph_outputs = [
        helper.make_tensor_value_info("output", ort_type, [batch_size, seq_len, hidden_size]),
        helper.make_tensor_value_info("present_key", ort_type, [batch_size, kv_num_heads, total_seq_len, head_size]),
        helper.make_tensor_value_info("present_value", ort_type, [batch_size, kv_num_heads, total_seq_len, head_size]),
    ]

    graph = helper.make_graph([node], "GQA_Graph", graph_inputs, graph_outputs)
    model = helper.make_model(graph)
    return model


def run_gqa_model(model, feed_dict):
    """Run the GQA model through ORT CPU and return outputs."""
    sess_options = SessionOptions()
    sess = InferenceSession(model.SerializeToString(), sess_options, providers=["CPUExecutionProvider"])
    outputs = sess.run(None, feed_dict)
    return outputs  # [output, present_key, present_value]


def np_to_list(arr):
    """Convert numpy array to a JSON-serializable list of Python floats/ints."""
    flat = arr.flatten().tolist()
    # Round floats for readability (avoid excessive decimals)
    if arr.dtype in (np.float32, np.float64):
        flat = [round(float(x), 6) if abs(x) > 1e-7 else 0.0 for x in flat]
    elif arr.dtype == np.float16:
        flat = [round(float(x), 4) if abs(float(x)) > 1e-4 else 0.0 for x in flat]
    return flat


def make_jsonc_tensor(data_np, ort_type, comment=None):
    """Create a JSONC tensor dict from a numpy array."""
    result = {}
    if comment:
        result["_comment"] = comment
    if data_np is None:
        result["data"] = None
        result["type"] = ort_type_to_jsonc_type(ort_type)
        return result
    result["data"] = np_to_list(data_np)
    result["dims"] = list(data_np.shape)
    result["type"] = ort_type_to_jsonc_type(
        TensorProto.INT32 if data_np.dtype == np.int32
        else TensorProto.INT64 if data_np.dtype == np.int64
        else ort_type
    )
    return result


class GQATestCase:
    """Encapsulates a single GQA test case config, generates data, runs ORT, and produces JSONC."""

    def __init__(
        self,
        name,
        batch_size,
        seq_len,
        num_heads,
        kv_num_heads,
        head_size,
        past_seq_len=0,
        packed=False,
        rotary=False,
        rotary_interleaved=False,
        softcap=0.0,
        smooth_softmax=False,
        local_window_size=-1,
        has_position_ids=False,
        has_attention_bias=False,
        has_head_sink=False,
        ort_type=TensorProto.FLOAT,
        seed=42,
    ):
        self.name = name
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.kv_num_heads = kv_num_heads
        self.head_size = head_size
        self.past_seq_len = past_seq_len
        self.packed = packed
        self.rotary = rotary
        self.rotary_interleaved = rotary_interleaved
        self.softcap = softcap
        self.smooth_softmax = smooth_softmax
        self.local_window_size = local_window_size
        self.has_position_ids = has_position_ids
        self.has_attention_bias = has_attention_bias
        self.has_head_sink = has_head_sink
        self.ort_type = ort_type
        self.seed = seed
        self.total_seq_len = past_seq_len + seq_len
        self.hidden_size = num_heads * head_size
        self.kv_hidden_size = kv_num_heads * head_size

    def generate_and_run(self):
        """Generate input data, run ORT, return (inputs_list, outputs_list) for JSONC."""
        np.random.seed(self.seed)
        dtype = ort_type_to_np_dtype(self.ort_type)

        # Generate input data
        query_dim = self.hidden_size if not self.packed else (self.hidden_size + 2 * self.kv_hidden_size)
        query = np.random.randn(self.batch_size, self.seq_len, query_dim).astype(dtype)

        if not self.packed:
            key = np.random.randn(self.batch_size, self.seq_len, self.kv_hidden_size).astype(dtype)
            value = np.random.randn(self.batch_size, self.seq_len, self.kv_hidden_size).astype(dtype)
        else:
            key = None
            value = None

        # Past KV in BNSH format
        if self.past_seq_len > 0:
            past_key = np.random.randn(self.batch_size, self.kv_num_heads, self.past_seq_len, self.head_size).astype(dtype)
            past_value = np.random.randn(self.batch_size, self.kv_num_heads, self.past_seq_len, self.head_size).astype(dtype)
        else:
            past_key = np.zeros((self.batch_size, self.kv_num_heads, 0, self.head_size), dtype=dtype)
            past_value = np.zeros((self.batch_size, self.kv_num_heads, 0, self.head_size), dtype=dtype)

        # seqlens_k is always total_sequence_length - 1 (per GQA op schema).
        # The kernel recovers total_sequence_length as seqlens_k + 1.
        seqlens_k = np.array([self.total_seq_len - 1] * self.batch_size, dtype=np.int32)

        total_seq_tensor = np.array([self.total_seq_len], dtype=np.int32)

        # Build feed dict
        feed_dict = {
            "query": query,
            "past_key": past_key,
            "past_value": past_value,
            "seqlens_k": seqlens_k,
            "total_sequence_length": total_seq_tensor,
        }
        if not self.packed:
            feed_dict["key"] = key
            feed_dict["value"] = value

        # Optional inputs
        cos_cache = None
        sin_cache = None
        if self.rotary:
            rotary_dim = (math.floor(self.head_size / 16) * 16) // 2
            angle = np.random.rand(self.total_seq_len, rotary_dim).astype(np.float64) * 2 * math.pi
            cos_cache = np.cos(angle).astype(dtype)
            sin_cache = np.sin(angle).astype(dtype)
            feed_dict["cos_cache"] = cos_cache
            feed_dict["sin_cache"] = sin_cache

        position_ids = None
        if self.has_position_ids:
            if self.past_seq_len > 0:
                # Token gen: positions start from past_seq_len
                position_ids = np.array(
                    [[self.past_seq_len + i for i in range(self.seq_len)] for _ in range(self.batch_size)],
                    dtype=np.int64,
                )
            else:
                # Prompt: positions are 0..seq_len-1
                position_ids = np.zeros((self.batch_size, self.seq_len), dtype=np.int64)
            feed_dict["position_ids"] = position_ids

        attention_bias = None
        if self.has_attention_bias:
            if self.past_seq_len > 0:
                # For token gen: create causal-like bias
                attention_bias = np.zeros((self.batch_size, 1, self.seq_len, self.total_seq_len), dtype=dtype)
            else:
                # For prompt: upper triangular bias
                bias = np.random.rand(self.batch_size, 1, self.seq_len, self.total_seq_len).astype(np.float64)
                bias = np.triu(bias, k=1).astype(dtype)
                attention_bias = bias
            feed_dict["attention_bias"] = attention_bias

        head_sink = None
        if self.has_head_sink:
            head_sink = np.random.rand(self.num_heads).astype(dtype)
            feed_dict["head_sink"] = head_sink

        # Build and run model
        model = create_gqa_onnx_model(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            num_heads=self.num_heads,
            kv_num_heads=self.kv_num_heads,
            head_size=self.head_size,
            past_seq_len=self.past_seq_len,
            packed=self.packed,
            rotary=self.rotary,
            rotary_interleaved=self.rotary_interleaved,
            softcap=self.softcap,
            smooth_softmax=self.smooth_softmax,
            local_window_size=self.local_window_size,
            has_position_ids=self.has_position_ids,
            has_attention_bias=self.has_attention_bias,
            has_head_sink=self.has_head_sink,
            ort_type=self.ort_type,
        )

        output, present_key, present_value = run_gqa_model(model, feed_dict)

        # Build JSONC inputs list (positional order matching ONNX schema)
        inputs = []
        # 0: query
        inputs.append(make_jsonc_tensor(query, self.ort_type))
        # 1: key
        if not self.packed:
            inputs.append(make_jsonc_tensor(key, self.ort_type))
        else:
            inputs.append({"data": None, "type": ort_type_to_jsonc_type(self.ort_type)})
        # 2: value
        if not self.packed:
            inputs.append(make_jsonc_tensor(value, self.ort_type))
        else:
            inputs.append({"data": None, "type": ort_type_to_jsonc_type(self.ort_type)})
        # 3: past_key (BNSH)
        inputs.append(make_jsonc_tensor(past_key, self.ort_type))
        # 4: past_value (BNSH)
        inputs.append(make_jsonc_tensor(past_value, self.ort_type))
        # 5: seqlens_k
        inputs.append(make_jsonc_tensor(seqlens_k, TensorProto.INT32))
        # 6: total_sequence_length
        inputs.append(make_jsonc_tensor(total_seq_tensor, TensorProto.INT32))
        # 7: cos_cache (optional)
        if self.rotary:
            inputs.append(make_jsonc_tensor(cos_cache, self.ort_type))
        # 8: sin_cache (optional)
        if self.rotary:
            inputs.append(make_jsonc_tensor(sin_cache, self.ort_type))
        # 9: position_ids (optional) - need null placeholders if before a present input
        if self.has_position_ids:
            # If rotary is not set, we need null entries for cos_cache and sin_cache
            if not self.rotary:
                inputs.append({"data": None, "type": ort_type_to_jsonc_type(self.ort_type)})  # cos_cache
                inputs.append({"data": None, "type": ort_type_to_jsonc_type(self.ort_type)})  # sin_cache
            inputs.append(make_jsonc_tensor(position_ids, TensorProto.INT64))
        # 10: attention_bias (optional)
        if self.has_attention_bias:
            # Fill null placeholders up to index 10 if needed
            needed_idx = 10
            while len(inputs) < needed_idx:
                # Determine what type the placeholder should be
                if len(inputs) in (7, 8):  # cos/sin cache
                    inputs.append({"data": None, "type": ort_type_to_jsonc_type(self.ort_type)})
                elif len(inputs) == 9:  # position_ids
                    inputs.append({"data": None, "type": "int64"})
                else:
                    inputs.append({"data": None, "type": ort_type_to_jsonc_type(self.ort_type)})
            inputs.append(make_jsonc_tensor(attention_bias, self.ort_type))
        # 11: head_sink (optional)
        if self.has_head_sink:
            needed_idx = 11
            while len(inputs) < needed_idx:
                if len(inputs) in (7, 8):
                    inputs.append({"data": None, "type": ort_type_to_jsonc_type(self.ort_type)})
                elif len(inputs) == 9:
                    inputs.append({"data": None, "type": "int64"})
                elif len(inputs) == 10:
                    inputs.append({"data": None, "type": ort_type_to_jsonc_type(self.ort_type)})
                else:
                    inputs.append({"data": None, "type": ort_type_to_jsonc_type(self.ort_type)})
            inputs.append(make_jsonc_tensor(head_sink, self.ort_type))

        # Build JSONC outputs list
        outputs = [
            make_jsonc_tensor(output, self.ort_type),
            make_jsonc_tensor(present_key, self.ort_type),
            make_jsonc_tensor(present_value, self.ort_type),
        ]

        return inputs, outputs

    def get_attributes(self):
        """Return the JSONC attributes list."""
        attrs = [
            {"name": "num_heads", "data": self.num_heads, "type": "int"},
            {"name": "kv_num_heads", "data": self.kv_num_heads, "type": "int"},
        ]
        if self.softcap != 0.0:
            attrs.append({"name": "softcap", "data": self.softcap, "type": "float"})
        if self.smooth_softmax:
            attrs.append({"name": "smooth_softmax", "data": 1, "type": "int"})
        if self.local_window_size != -1:
            attrs.append({"name": "local_window_size", "data": self.local_window_size, "type": "int"})
        if self.rotary:
            attrs.append({"name": "do_rotary", "data": 1, "type": "int"})
        if self.rotary_interleaved:
            attrs.append({"name": "rotary_interleaved", "data": 1, "type": "int"})
        return attrs


def generate_jsonc_file(test_cases, output_path):
    """Generate a JSONC file from a list of GQATestCase objects."""
    all_tests = []

    for tc in test_cases:
        print(f"  Generating: {tc.name}...")
        inputs, outputs = tc.generate_and_run()

        test_entry = {
            "name": tc.name,
            "operator": "GroupQueryAttention",
            "opset": {"domain": "com.microsoft", "version": 1},
            "attributes": tc.get_attributes(),
            "cases": [
                {
                    "name": "T[0]",
                    "inputs": inputs,
                    "outputs": outputs,
                }
            ],
        }
        all_tests.append(test_entry)

    # Write as JSON (JSONC is just JSON with comments; we output valid JSON)
    with open(output_path, "w", newline="\n") as f:
        json.dump(all_tests, f, indent=2)

    print(f"  Written: {output_path}")
    return output_path
