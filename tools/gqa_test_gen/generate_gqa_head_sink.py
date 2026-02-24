"""Generate GQA test data for head_sink feature.

head_sink follows the pattern in test_gqa_cpu.py:
- head_sink works with smooth_softmax (smooth_softmax_ref is called when head_sink is not None)
- head_sink is a 1D tensor of shape (num_heads)
- In test_gqa_cpu.py: smooth_softmax and head_sink are mutually exclusive
  (if use_smooth_softmax and head_sink: continue)
  So head_sink tests do NOT set smooth_softmax=1 attribute; the kernel
  internally handles the smooth softmax logic when head_sink is provided.

NOTE: The ONNX kernel's behavior is: when head_sink input is provided,
the kernel uses smooth softmax with the head_sink values. When smooth_softmax
attribute is set, the kernel uses smooth softmax without head_sink values (head_sink=None).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from gqa_test_utils import GQATestCase, generate_jsonc_file

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "js", "web", "test", "data", "ops")


def main():
    test_cases = [
        # Test 0: Simple prompt with head_sink, B=1, S=1, num_heads=1
        GQATestCase(
            name="GQA head_sink 0: prompt S=1",
            batch_size=1, seq_len=1, num_heads=1, kv_num_heads=1, head_size=8,
            past_seq_len=0, has_head_sink=True, seed=300,
        ),
        # Test 1: Prompt with head_sink, multi-token
        GQATestCase(
            name="GQA head_sink 1: prompt S=2",
            batch_size=1, seq_len=2, num_heads=1, kv_num_heads=1, head_size=8,
            past_seq_len=0, has_head_sink=True, seed=301,
        ),
        # Test 2: Token gen with past and head_sink
        GQATestCase(
            name="GQA head_sink 2: past S=1 past=2",
            batch_size=1, seq_len=1, num_heads=1, kv_num_heads=1, head_size=8,
            past_seq_len=2, has_head_sink=True, seed=302,
        ),
        # Test 3: GQA grouping with head_sink, num_heads=2
        GQATestCase(
            name="GQA head_sink 3: GQA grouping",
            batch_size=1, seq_len=2, num_heads=2, kv_num_heads=1, head_size=8,
            past_seq_len=0, has_head_sink=True, seed=303,
        ),
        # Test 4: head_sink with past and GQA grouping
        GQATestCase(
            name="GQA head_sink 4: past GQA grouping",
            batch_size=1, seq_len=1, num_heads=2, kv_num_heads=1, head_size=8,
            past_seq_len=2, has_head_sink=True, seed=304,
        ),
    ]

    output_path = os.path.join(OUTPUT_DIR, "group-query-attention-head-sink.jsonc")
    generate_jsonc_file(test_cases, output_path)


if __name__ == "__main__":
    main()
