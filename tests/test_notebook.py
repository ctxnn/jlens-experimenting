import json
import unittest
from pathlib import Path


class NotebookContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(Path("notebooks/qwen_official_jlens.ipynb").read_text())
        cls.source = "\n".join(
            "".join(cell.get("source", [])) for cell in cls.notebook["cells"]
        )

    def test_official_pinned_gpu_experiment(self):
        self.assertEqual(self.notebook["metadata"]["accelerator"], "GPU")
        self.assertIn("anthropics/jacobian-lens.git@581d398", self.source)
        self.assertIn("assert torch.cuda.is_available()", self.source)

    def test_published_qwen_lens_without_fitting(self):
        self.assertIn('MODEL_NAME = "Qwen/Qwen3.5-4B"', self.source)
        self.assertIn("Qwen3.5-4B_jacobian_lens_n1000.pt", self.source)
        self.assertNotIn("jlens.fit(", self.source)

    def test_completion_readout_uses_final_supplied_position(self):
        self.assertIn("READOUT_POSITION = -1", self.source)
        self.assertIn("positions=[READOUT_POSITION]", self.source)
        self.assertIn("logits[0, READOUT_POSITION]", self.source)
        self.assertIn('"readout_position": READOUT_POSITION', self.source)
        self.assertNotIn("positions=[-2]", self.source)
        self.assertNotIn("logits[0, -2]", self.source)

    def test_position_semantics_are_explained(self):
        self.assertIn("position `p` predict token `p + 1`", self.source)
        self.assertIn("predicts next unseen token", self.source)
        self.assertIn("teacher-forced reconstruction", self.source)

    def test_controls_metrics_and_visualization(self):
        expected_fragments = (
            "use_jacobian=False",
            "torch.allclose(direct, final",
            "top10_overlap",
            "compute_slice(",
            "build_page(",
            "qwen_validation.json",
        )
        for expected in expected_fragments:
            self.assertIn(expected, self.source)


class DocumentationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.understanding = Path("understanding.md").read_text()
        cls.change_log = Path("changes_log.md").read_text()
        cls.readme = Path("README.md").read_text()

    def test_readme_links_long_form_documentation(self):
        self.assertIn("[`understanding.md`](understanding.md)", self.readme)
        self.assertIn("[`changes_log.md`](changes_log.md)", self.readme)

    def test_understanding_documents_experiment_contract(self):
        required = (
            "Experiment version:** 0.3.0",
            "READOUT_POSITION = -1",
            "```mermaid",
            "Generated artifacts and schema",
            "Validation gates and what each proves",
            "Failure modes and troubleshooting",
        )
        for fragment in required:
            self.assertIn(fragment, self.understanding)

    def test_change_log_records_versions_and_position_fix(self):
        for version in ("[0.3.0]", "[0.2.0]", "[0.1.0]"):
            self.assertIn(version, self.change_log)
        self.assertIn("teacher-forced reconstruction", self.change_log)
        self.assertIn("positions=[-2]", self.change_log)


if __name__ == "__main__":
    unittest.main()
