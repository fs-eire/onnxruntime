"""Generate GQA test data for attention_bias feature.

attention_bias has shape (batch_size or 1, num_heads or 1, sequence_length, total_sequence_length).
It's added to QK^T before softmax.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from gqa_test_utils import GQATestCase, generate_jsonc_file

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "js", "web", "test", "data", "ops")


def main():
    test_cases = [
        # Test 0: Prompt with attention_bias
        GQATestCase(
            name="GQA attention_bias 0: prompt S=1",
            batch_size=1, seq_len=1, num_heads=1, kv_num_heads=1, head_size=8,
            past_seq_len=0, has_attention_bias=True, seed=800,
        ),
        # Test 1: Prompt multi-token
        GQATestCase(
            name="GQA attention_bias 1: prompt S=3",
            batch_size=1, seq_len=3, num_heads=1, kv_num_heads=1, head_size=8,
            past_seq_len=0, has_attention_bias=True, seed=801,
        ),
        # Test 2: Token gen with past
        GQATestCase(
            name="GQA attention_bias 2: past S=1 past=2",
            batch_size=1, seq_len=1, num_heads=1, kv_num_heads=1, head_size=8,
            past_seq_len=2, has_attention_bias=True, seed=802,
        ),
        # Test 3: GQA grouping with attention_bias
        GQATestCase(
            name="GQA attention_bias 3: GQA grouping",
            batch_size=1, seq_len=2, num_heads=2, kv_num_heads=1, head_size=8,
            past_seq_len=0, has_attention_bias=True, seed=803,
        ),
        # Test 4: Token gen with past and GQA grouping
        GQATestCase(
            name="GQA attention_bias 4: past GQA",
            batch_size=1, seq_len=1, num_heads=2, kv_num_heads=1, head_size=8,
            past_seq_len=2, has_attention_bias=True, seed=804,
        ),
    ]

    output_path = os.path.join(OUTPUT_DIR, "group-query-attention-attention-bias.jsonc")
    generate_jsonc_file(test_cases, output_path)


if __name__ == "__main__":
    main()
