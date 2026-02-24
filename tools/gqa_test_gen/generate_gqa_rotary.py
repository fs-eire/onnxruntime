"""Generate GQA test data for do_rotary feature (non-interleaved).

Rotary requires head_size to be a multiple of 16.
cos_cache and sin_cache have shape (max_seq_len, head_size/2).
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
            name="GQA rotary 0: prompt S=1",
            batch_size=1, seq_len=1, num_heads=1, kv_num_heads=1, head_size=16,
            past_seq_len=0, rotary=True, seed=400,
        ),
        # Test 1: Prompt, multi-token
        GQATestCase(
            name="GQA rotary 1: prompt S=3",
            batch_size=1, seq_len=3, num_heads=1, kv_num_heads=1, head_size=16,
            past_seq_len=0, rotary=True, seed=401,
        ),
        # Test 2: Token gen with past
        GQATestCase(
            name="GQA rotary 2: past S=1 past=2",
            batch_size=1, seq_len=1, num_heads=1, kv_num_heads=1, head_size=16,
            past_seq_len=2, rotary=True, seed=402,
        ),
        # Test 3: GQA grouping with rotary
        GQATestCase(
            name="GQA rotary 3: GQA grouping",
            batch_size=1, seq_len=2, num_heads=2, kv_num_heads=1, head_size=16,
            past_seq_len=0, rotary=True, seed=403,
        ),
        # Test 4: Token gen, GQA grouping, larger head_size
        GQATestCase(
            name="GQA rotary 4: past GQA grouping h=32",
            batch_size=1, seq_len=1, num_heads=2, kv_num_heads=1, head_size=32,
            past_seq_len=2, rotary=True, seed=404,
        ),
    ]

    output_path = os.path.join(OUTPUT_DIR, "group-query-attention-rotary.jsonc")
    generate_jsonc_file(test_cases, output_path)


if __name__ == "__main__":
    main()
