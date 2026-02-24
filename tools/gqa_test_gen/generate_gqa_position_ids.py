"""Generate GQA test data for position_ids feature.

position_ids is used with rotary embeddings to specify the position for each token.
- Shape: (batch_size, sequence_length), dtype: int64
- For prompt: typically all zeros (kernel uses first element)
- For token gen: past_seq_len + i for each position i
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from gqa_test_utils import GQATestCase, generate_jsonc_file

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "js", "web", "test", "data", "ops")


def main():
    test_cases = [
        # Test 0: Prompt with position_ids + rotary
        GQATestCase(
            name="GQA position_ids 0: prompt S=1",
            batch_size=1, seq_len=1, num_heads=1, kv_num_heads=1, head_size=16,
            past_seq_len=0, rotary=True, has_position_ids=True, seed=700,
        ),
        # Test 1: Prompt multi-token
        GQATestCase(
            name="GQA position_ids 1: prompt S=3",
            batch_size=1, seq_len=3, num_heads=1, kv_num_heads=1, head_size=16,
            past_seq_len=0, rotary=True, has_position_ids=True, seed=701,
        ),
        # Test 2: Token gen with past
        GQATestCase(
            name="GQA position_ids 2: past S=1 past=2",
            batch_size=1, seq_len=1, num_heads=1, kv_num_heads=1, head_size=16,
            past_seq_len=2, rotary=True, has_position_ids=True, seed=702,
        ),
        # Test 3: GQA grouping with position_ids
        GQATestCase(
            name="GQA position_ids 3: GQA grouping",
            batch_size=1, seq_len=2, num_heads=2, kv_num_heads=1, head_size=16,
            past_seq_len=0, rotary=True, has_position_ids=True, seed=703,
        ),
        # Test 4: Token gen, GQA grouping
        GQATestCase(
            name="GQA position_ids 4: past GQA grouping",
            batch_size=1, seq_len=1, num_heads=2, kv_num_heads=1, head_size=16,
            past_seq_len=2, rotary=True, has_position_ids=True, seed=704,
        ),
    ]

    output_path = os.path.join(OUTPUT_DIR, "group-query-attention-position-ids.jsonc")
    generate_jsonc_file(test_cases, output_path)


if __name__ == "__main__":
    main()
