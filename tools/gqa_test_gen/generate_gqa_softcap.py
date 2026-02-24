"""Generate GQA test data for softcap feature."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from gqa_test_utils import GQATestCase, generate_jsonc_file

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "js", "web", "test", "data", "ops")


def main():
    test_cases = [
        # Test 0: Simple prompt, no past, B=1, S=1, softcap=50.0
        GQATestCase(
            name="GQA softcap 0: prompt S=1",
            batch_size=1, seq_len=1, num_heads=1, kv_num_heads=1, head_size=8,
            past_seq_len=0, softcap=50.0, seed=100,
        ),
        # Test 1: Prompt, no past, B=1, S=3, softcap=50.0
        GQATestCase(
            name="GQA softcap 1: prompt S=3",
            batch_size=1, seq_len=3, num_heads=1, kv_num_heads=1, head_size=8,
            past_seq_len=0, softcap=50.0, seed=101,
        ),
        # Test 2: Token gen with past, B=1, S=1, past=2, softcap=50.0
        GQATestCase(
            name="GQA softcap 2: past S=1 past=2",
            batch_size=1, seq_len=1, num_heads=1, kv_num_heads=1, head_size=8,
            past_seq_len=2, softcap=50.0, seed=102,
        ),
        # Test 3: GQA grouping with softcap, B=1, S=2, num_heads=2, kv_num_heads=1
        GQATestCase(
            name="GQA softcap 3: GQA grouping",
            batch_size=1, seq_len=2, num_heads=2, kv_num_heads=1, head_size=8,
            past_seq_len=0, softcap=50.0, seed=103,
        ),
        # Test 4: Different softcap value, with past
        GQATestCase(
            name="GQA softcap 4: softcap=10 past=1",
            batch_size=1, seq_len=1, num_heads=2, kv_num_heads=1, head_size=8,
            past_seq_len=1, softcap=10.0, seed=104,
        ),
    ]

    output_path = os.path.join(OUTPUT_DIR, "group-query-attention-softcap.jsonc")
    generate_jsonc_file(test_cases, output_path)


if __name__ == "__main__":
    main()
