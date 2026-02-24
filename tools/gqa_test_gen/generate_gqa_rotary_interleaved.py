"""Generate GQA test data for rotary_interleaved feature.

rotary_interleaved=1 means the rotary embedding uses interleaved pattern
(x[0::2] and x[1::2]) instead of split-half pattern.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from gqa_test_utils import GQATestCase, generate_jsonc_file

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "js", "web", "test", "data", "ops")


def main():
    test_cases = [
        # Test 0: Prompt, no past, head_size=16
        GQATestCase(
            name="GQA rotary_interleaved 0: prompt S=1",
            batch_size=1, seq_len=1, num_heads=1, kv_num_heads=1, head_size=16,
            past_seq_len=0, rotary=True, rotary_interleaved=True, seed=500,
        ),
        # Test 1: Prompt, multi-token
        GQATestCase(
            name="GQA rotary_interleaved 1: prompt S=3",
            batch_size=1, seq_len=3, num_heads=1, kv_num_heads=1, head_size=16,
            past_seq_len=0, rotary=True, rotary_interleaved=True, seed=501,
        ),
        # Test 2: Token gen with past
        GQATestCase(
            name="GQA rotary_interleaved 2: past S=1 past=2",
            batch_size=1, seq_len=1, num_heads=1, kv_num_heads=1, head_size=16,
            past_seq_len=2, rotary=True, rotary_interleaved=True, seed=502,
        ),
        # Test 3: GQA grouping
        GQATestCase(
            name="GQA rotary_interleaved 3: GQA grouping",
            batch_size=1, seq_len=2, num_heads=2, kv_num_heads=1, head_size=16,
            past_seq_len=0, rotary=True, rotary_interleaved=True, seed=503,
        ),
        # Test 4: Token gen with past, GQA grouping, h=32
        GQATestCase(
            name="GQA rotary_interleaved 4: past GQA h=32",
            batch_size=1, seq_len=1, num_heads=2, kv_num_heads=1, head_size=32,
            past_seq_len=2, rotary=True, rotary_interleaved=True, seed=504,
        ),
    ]

    output_path = os.path.join(OUTPUT_DIR, "group-query-attention-rotary-interleaved.jsonc")
    generate_jsonc_file(test_cases, output_path)


if __name__ == "__main__":
    main()
