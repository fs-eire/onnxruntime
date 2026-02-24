"""Generate GQA test data for local_window_size (sliding window attention).

local_window_size limits how far back each query can attend.
Need multi-token sequences to see the windowing effect.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from gqa_test_utils import GQATestCase, generate_jsonc_file

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "js", "web", "test", "data", "ops")


def main():
    test_cases = [
        # Test 0: Prompt S=4, window=2 (each token sees at most 2 previous + itself)
        GQATestCase(
            name="GQA local_window 0: prompt S=4 w=2",
            batch_size=1, seq_len=4, num_heads=1, kv_num_heads=1, head_size=8,
            past_seq_len=0, local_window_size=2, seed=600,
        ),
        # Test 1: Prompt S=3, window=1
        GQATestCase(
            name="GQA local_window 1: prompt S=3 w=1",
            batch_size=1, seq_len=3, num_heads=1, kv_num_heads=1, head_size=8,
            past_seq_len=0, local_window_size=1, seed=601,
        ),
        # Test 2: Token gen with past, window=2
        GQATestCase(
            name="GQA local_window 2: past=3 S=1 w=2",
            batch_size=1, seq_len=1, num_heads=1, kv_num_heads=1, head_size=8,
            past_seq_len=3, local_window_size=2, seed=602,
        ),
        # Test 3: GQA grouping with window
        GQATestCase(
            name="GQA local_window 3: GQA grouping S=4 w=2",
            batch_size=1, seq_len=4, num_heads=2, kv_num_heads=1, head_size=8,
            past_seq_len=0, local_window_size=2, seed=603,
        ),
        # Test 4: Token gen with past, GQA grouping, window=1
        GQATestCase(
            name="GQA local_window 4: past GQA w=1",
            batch_size=1, seq_len=1, num_heads=2, kv_num_heads=1, head_size=8,
            past_seq_len=3, local_window_size=1, seed=604,
        ),
    ]

    output_path = os.path.join(OUTPUT_DIR, "group-query-attention-local-window.jsonc")
    generate_jsonc_file(test_cases, output_path)


if __name__ == "__main__":
    main()
