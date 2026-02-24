"""Run all GQA test data generation scripts."""
import subprocess
import sys
import os

SCRIPTS_DIR = os.path.dirname(__file__)
PYTHON = sys.executable

scripts = [
    "generate_gqa_softcap.py",
    "generate_gqa_smooth_softmax.py",
    "generate_gqa_head_sink.py",
    "generate_gqa_rotary.py",
    "generate_gqa_rotary_interleaved.py",
    "generate_gqa_local_window.py",
    "generate_gqa_position_ids.py",
    "generate_gqa_attention_bias.py",
    "generate_gqa_float16.py",
]

failed = []
for script in scripts:
    path = os.path.join(SCRIPTS_DIR, script)
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print(f"{'='*60}")
    result = subprocess.run([PYTHON, path], capture_output=False)
    if result.returncode != 0:
        failed.append(script)
        print(f"  FAILED with return code {result.returncode}")
    else:
        print(f"  OK")

if failed:
    print(f"\n{'='*60}")
    print(f"FAILED scripts: {failed}")
    sys.exit(1)
else:
    print(f"\n{'='*60}")
    print("All scripts completed successfully!")
