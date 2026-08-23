import json
import unittest
from pathlib import Path

class NotebookContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(Path("notebooks/qwen_official_jlens.ipynb").read_text())
        cls.source = "\n".join("".join(c.get("source", [])) for c in cls.notebook["cells"])

    def test_official_pinned_gpu_experiment(self):
        self.assertEqual(self.notebook["metadata"]["accelerator"], "GPU")
        self.assertIn("anthropics/jacobian-lens.git@581d398", self.source)
        self.assertIn("assert torch.cuda.is_available()", self.source)

    def test_published_qwen_lens_without_fitting(self):
        self.assertIn('MODEL_NAME = "Qwen/Qwen3.5-4B"', self.source)
        self.assertIn("Qwen3.5-4B_jacobian_lens_n1000.pt", self.source)
        self.assertNotIn("jlens.fit(", self.source)

    def test_controls_metrics_and_visualization(self):
        for expected in ("use_jacobian=False", "torch.allclose(direct, final", "top10_overlap", "compute_slice(", "build_page(", "qwen_validation.json"):
            self.assertIn(expected, self.source)

if __name__ == "__main__": unittest.main()
