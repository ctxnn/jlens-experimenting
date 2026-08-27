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

    def test_all_cells_have_unique_stable_ids(self):
        ids = [cell.get("id") for cell in self.notebook["cells"]]
        self.assertEqual(len(ids), 11)
        self.assertTrue(all(ids))
        self.assertEqual(len(ids), len(set(ids)))

    def test_model_and_tokenizer_use_immutable_revision(self):
        revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        self.assertIn(f'MODEL_REVISION = "{revision}"', self.source)
        self.assertEqual(self.source.count("revision=MODEL_REVISION"), 3)
        self.assertIn("hf_hub_download(", self.source)
        self.assertIn('resolved_config_path.parts.index("snapshots")', self.source)
        self.assertIn("RESOLVED_MODEL_REVISION == MODEL_REVISION", self.source)
        self.assertNotIn("_commit_hash", self.source)
        self.assertIn('"model_revision": MODEL_REVISION', self.source)
        self.assertIn('"resolved_model_revision": RESOLVED_MODEL_REVISION', self.source)

    def test_lens_uses_immutable_revision(self):
        revision = "16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a"
        self.assertIn(f'LENS_REVISION = "{revision}"', self.source)
        self.assertIn('LENS_SOURCE_REF = "qwen-n1000"', self.source)
        self.assertIn("revision=LENS_REVISION", self.source)
        self.assertIn('"lens_revision": LENS_REVISION', self.source)
        self.assertIn('"lens_source_ref": LENS_SOURCE_REF', self.source)
        self.assertNotIn('LENS_REVISION = "qwen-n1000"', self.source)

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
        cls.atlas = Path("visualizations/jlens-concept-atlas.html").read_text()

    def test_readme_links_long_form_documentation(self):
        self.assertIn("[`understanding.md`](understanding.md)", self.readme)
        self.assertIn("[`changes_log.md`](changes_log.md)", self.readme)

    def test_readme_documents_colab_cli_and_error_audit(self):
        self.assertIn("colab exec -s jlens-qwen", self.readme)
        self.assertIn('select(.output_type == "error")', self.readme)
        self.assertIn("colab stop -s jlens-qwen", self.readme)
        self.assertIn("## Iteration 1 verification", self.readme)
        self.assertIn("maximum final-readout parity error was `0.03125`", self.readme)

    def test_understanding_documents_experiment_contract(self):
        required = (
            "Experiment version:** 0.5.0",
            "Lens revision | `16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a`",
            "Model revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`",
            "READOUT_POSITION = -1",
            "```mermaid",
            "Generated artifacts and schema",
            "Validation gates and what each proves",
            "Failure modes and troubleshooting",
            '"resolved_model_revision"',
            "Iteration 1 live verification",
            "This closes the visual-QA follow-up",
            "committed as an explicit",
        )
        for fragment in required:
            self.assertIn(fragment, self.understanding)

    def test_change_log_records_versions_and_position_fix(self):
        for version in ("[0.5.0]", "[0.4.0]", "[0.3.3]", "[0.3.2]", "[0.3.1]", "[0.3.0]", "[0.2.0]", "[0.1.0]"):
            self.assertIn(version, self.change_log)
        self.assertIn("teacher-forced reconstruction", self.change_log)
        self.assertIn("positions=[-2]", self.change_log)
        self.assertIn("model_revision", self.change_log)
        self.assertIn("lens_source_ref", self.change_log)

    def test_concept_atlas_preserves_scientific_distinctions(self):
        self.assertIn("visualizations/jlens-concept-atlas.html", self.readme)
        self.assertIn("J-Lens is a transport, not a label maker", self.atlas)
        self.assertIn("W<sub>U</sub> · N(J<sub>ℓ</sub> h<sub>ℓ,p</sub>)", self.atlas)
        self.assertIn("W<sub>U</sub> · N(h<sub>ℓ,p</sub>)", self.atlas)
        self.assertIn("not confidence or accuracy", self.atlas)
        self.assertIn("READOUT_POSITION = -1", self.atlas)
        self.assertIn("not proof that the concept caused the answer", self.atlas)


class RelaySuiteContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = Path("experiments/qwen_relay_suite.py").read_text()
        cls.data = json.loads(Path("artifacts/qwen_relay_suite.json").read_text())
        cls.summary = json.loads(
            Path("artifacts/qwen_relay_suite_summary.json").read_text()
        )
        cls.visualization = Path(
            "visualizations/qwen-concept-relay.html"
        ).read_text()
        cls.readme = Path("README.md").read_text()
        cls.understanding = Path("understanding.md").read_text()
        cls.change_log = Path("changes_log.md").read_text()

    def test_suite_is_pinned_and_explicitly_non_causal(self):
        self.assertIn('MODEL_NAME = "Qwen/Qwen3.5-4B"', self.experiment)
        self.assertIn(
            'MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"',
            self.experiment,
        )
        self.assertIn(
            'LENS_REVISION = "16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a"',
            self.experiment,
        )
        self.assertEqual(self.data["scope"], "readout-only; no causal interventions")

    def test_suite_retains_the_complete_prompt_population(self):
        results = self.data["results"]
        ids = [row["id"] for row in results]
        self.assertEqual(len(results), 63)
        self.assertEqual(len(set(ids)), 63)
        self.assertEqual(
            {condition: sum(row["condition"] == condition for row in results)
             for condition in ("hidden", "explicit")},
            {"hidden": 51, "explicit": 12},
        )
        self.assertTrue(all(len(row["layers"]) == 31 for row in results))
        self.assertLessEqual(
            max(row["model_parity_max_abs_error"] for row in results), 0.125
        )

    def test_observed_aggregate_is_reconciled(self):
        aggregate = self.data["aggregate"]
        self.assertEqual(
            {key: aggregate[key] for key in ("total", "correct", "measurable", "relay", "target_specific")},
            {"total": 63, "correct": 50, "measurable": 46, "relay": 7, "target_specific": 48},
        )
        self.assertEqual(sum(row["summary"]["relay"] for row in self.data["results"]), 7)
        self.assertEqual(sum(row["correct"] for row in self.data["results"]), 50)
        self.assertEqual(sum(row["measurable"] for row in self.data["results"]), 46)

    def test_compact_summary_matches_raw_results(self):
        self.assertEqual(self.summary["aggregate"], self.data["aggregate"])
        self.assertEqual(
            [row["id"] for row in self.summary["results"]],
            [row["id"] for row in self.data["results"]],
        )
        self.assertTrue(all(len(row["layers"]) == 31 for row in self.summary["results"]))

    def test_visualization_exposes_failures_and_interpretation_boundary(self):
        required = (
            "Concept relay across 63 prompts",
            "All prompts, no cherry-pick",
            "incorrect",
            "token-split",
            "logit lens",
            "not a causal intervention",
            "not a literal transcript",
            "animal-spider-web-catcher",
            "capital-france",
        )
        for fragment in required:
            self.assertIn(fragment, self.visualization)
        self.assertNotIn("__QWEN_RELAY_DATA__", self.visualization)

    def test_release_documentation_records_observed_suite(self):
        self.assertIn(
            "visualizations/qwen-concept-relay.html", self.readme
        )
        self.assertIn("50 / 63", self.readme)
        self.assertIn("Experiment version:** 0.5.0", self.understanding)
        self.assertIn("## 10. Multi-prompt relay verification", self.understanding)
        self.assertIn("## [0.5.0]", self.change_log)
        self.assertIn("17 token-split rows", self.change_log)


if __name__ == "__main__":
    unittest.main()
