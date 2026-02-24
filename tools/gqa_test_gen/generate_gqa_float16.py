"""Generate GQA test data for float16 type.

Same basic GQA patterns but with float16 data type.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from onnx import TensorProto
from gqa_test_utils import GQATestCase, generate_jsonc_file

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "js", "web", "test", "data", "ops")


def main():
    test_cases = [
        # Test 0: Simple prompt, no past, float16
        GQATestCase(
            name="GQA float16 0: prompt S=1",
            batch_size=1, seq_len=1, num_heads=1, kv_num_heads=1, head_size=8,
            past_seq_len=0, ort_type=TensorProto.FLOAT16, seed=900,
        ),
        # Test 1: Prompt multi-token, float16
        GQATestCase(
            name="GQA float16 1: prompt S=3",
            batch_size=1, seq_len=3, num_heads=1, kv_num_heads=1, head_size=8,
            past_seq_len=0, ort_type=TensorProto.FLOAT16, seed=901,
        ),
        # Test 2: Token gen with past, float16
        GQATestCase(
            name="GQA float16 2: past S=1 past=2",
            batch_size=1, seq_len=1, num_heads=1, kv_num_heads=1, head_size=8,
            past_seq_len=2, ort_type=TensorProto.FLOAT16, seed=902,
        ),
        # Test 3: GQA grouping, float16
        GQATestCase(
            name="GQA float16 3: GQA grouping",
            batch_size=1, seq_len=2, num_heads=2, kv_num_heads=1, head_size=8,
            past_seq_len=0, ort_type=TensorProto.FLOAT16, seed=903,
        ),
        # Test 4: Token gen with past and GQA grouping, float16
        GQATestCase(
            name="GQA float16 4: past GQA grouping",
            batch_size=1, seq_len=1, num_heads=2, kv_num_heads=1, head_size=8,
            past_seq_len=2, ort_type=TensorProto.FLOAT16, seed=904,
        ),
    ]

    output_path = os.path.join(OUTPUT_DIR, "group-query-attention-float16.jsonc")
    generate_jsonc_file(test_cases, output_path)


if __name__ == "__main__":
    main()
