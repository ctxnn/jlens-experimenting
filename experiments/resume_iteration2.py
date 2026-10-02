"""Colab companion cell: resume the frozen 20-case suite after inspecting the pilot.

Execute with `colab exec -s SESSION -f experiments/resume_iteration2.py --timeout 1800`.
The uploaded runner and protocol must already exist in /content.
"""
import os
from pathlib import Path

os.environ["I2_CASE_LIMIT"] = "20"
os.environ["I2_PROTOCOL"] = "/content/iteration2_prompts.json"
os.environ["I2_OUTPUT"] = "/content/qwen_iteration2.json"
exec(compile(Path("/content/qwen_iteration2.py").read_text(), "/content/qwen_iteration2.py", "exec"), globals())
