"""Colab pilot cell for the paired hidden-description pass (first three cases)."""
import os
from pathlib import Path

os.environ["I2_PROTOCOL"] = "/content/iteration2_hidden_prompts.json"
os.environ["I2_OUTPUT"] = "/content/qwen_iteration2_hidden.json"
os.environ["I2_CASE_LIMIT"] = "3"
exec(compile(Path("/content/qwen_iteration2.py").read_text(), "/content/qwen_iteration2.py", "exec"), globals())
