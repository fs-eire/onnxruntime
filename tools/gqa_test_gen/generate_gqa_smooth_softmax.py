"""Generate GQA test data for smooth_softmax feature."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from gqa_test_utils import GQATestCase, generate_jsonc_file

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "js", "web", "test", "data", "ops")


def main():
    test_cases = [
        # Test 0: Simple prompt, no past, B=1, S=1
        GQATestCase(
            name="GQA smooth_softmax 0: prompt S=1",
            batch_size=1, seq_len=1, num_heads=1, kv_num_heads=1, head_size=8,
            past_seq_len=0, smooth_softmax=True, seed=200,
        ),
        # Test 1: Prompt, S=2, multi-token to see softmax difference
        GQATestCase(
            name="GQA smooth_softmax 1: prompt S=2",
            batch_size=1, seq_len=2, num_heads=1, kv_num_heads=1, head_size=8,
            past_seq_len=0, smooth_softmax=True, seed=201,
        ),
        # Test 2: Token gen with past
        GQATestCase(
            name="GQA smooth_softmax 2: past S=1 past=2",
            batch_size=1, seq_len=1, num_heads=1, kv_num_heads=1, head_size=8,
            past_seq_len=2, smooth_softmax=True, seed=202,
        ),
        # Test 3: GQA grouping
        GQATestCase(
            name="GQA smooth_softmax 3: GQA grouping",
            batch_size=1, seq_len=2, num_heads=2, kv_num_heads=1, head_size=8,
            past_seq_len=0, smooth_softmax=True, seed=203,
        ),
        # Test 4: With past and GQA grouping
        GQATestCase(
            name="GQA smooth_softmax 4: past GQA grouping",
            batch_size=1, seq_len=1, num_heads=2, kv_num_heads=1, head_size=8,
            past_seq_len=2, smooth_softmax=True, seed=204,
        ),
    ]

    output_path = os.path.join(OUTPUT_DIR, "group-query-attention-smooth-softmax.jsonc")
    generate_jsonc_file(test_cases, output_path)


if __name__ == "__main__":
    main()
