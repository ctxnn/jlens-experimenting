"""Release-caption and redistribution regressions; not browser-rendered QA."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "figure_builder", ROOT / "experiments/build_iteration2_visualization.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
NS = {"s": "http://www.w3.org/2000/svg"}


class PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = []
        for name in ("qwen_iteration2_hidden.json", "qwen_iteration2.json"):
            cls.cases.extend(json.loads((ROOT / "artifacts/iteration2" / name).read_text())["cases"])

    def test_every_intervention_export_keeps_interpretation_context(self):
        for case in self.cases:
            source = (ROOT / f'visualizations/iteration2/{case["id"]}-intervention.svg').read_text()
            figure = ET.fromstring(source)
            texts = [element.text or "" for element in figure.findall(".//s:text", NS)]
            self.assertIn("Next-token alias mass, not whole-answer probability · fixed 0–100% axis", texts)
            self.assertTrue(any("one forward over the supplied prefix" in text for text in texts))
            self.assertTrue(any("not an answer-swap success" in text for text in texts))
            clean = re.sub(r"\s+", " ", case["clean_generation"]["text"]).strip()
            clean = clean if len(clean) <= 96 else clean[:95] + "…"
            self.assertIn(f'Clean continuation: “{clean}”', texts)
            row = next(r for r in case["interventions"]
                       if r["layer"] == 24 and r["condition"] == "concept_swap")
            if row["available"]:
                edited = re.sub(r"\s+", " ", row["generation"]["text"]).strip()
                edited = edited if len(edited) <= 96 else edited[:95] + "…"
                self.assertIn(f'Concept-edit continuation: “{edited}”', texts)
                self.assertTrue(any("sustained at layer 24 on every decoding forward" in t for t in texts))
            else:
                self.assertIn("Concept-edit continuation: unavailable (no single-token concept pair)", texts)

    def test_spider_exports_do_not_imply_the_intended_six_answer(self):
        for case_id, edited_opening in (("spider-legs-hidden", "8."), ("spider-quiz-hidden", "4 ")):
            root = ET.parse(ROOT / f"visualizations/iteration2/{case_id}-intervention.svg")
            texts = [e.text or "" for e in root.findall(".//s:text", NS)]
            self.assertTrue(any(t.startswith('Concept-edit continuation: “' + edited_opening) for t in texts))

    def test_other_layer_export_does_not_mislabel_fixed_layer_generation(self):
        svg = BUILDER.intervention_svg(self.cases[0], 8)
        self.assertIn("not recorded at this layer", svg)
        self.assertNotIn("sustained at layer 8", svg)

    def test_current_figures_are_deterministic(self):
        for case in self.cases:
            for suffix, make in (("depth", BUILDER.trace_svg), ("intervention", BUILDER.intervention_svg)):
                saved = (ROOT / f'visualizations/iteration2/{case["id"]}-{suffix}.svg').read_text()
                self.assertEqual(saved, make(case), case["id"] + "/" + suffix)

    def test_raw_measurements_remain_unchanged(self):
        for name, digest in (
            ("qwen_iteration2.json", "1e1f39b557157ed1fb32505b8ffe87f2ab4d630a1c0d32bfb8e9da23643b7e0d"),
            ("qwen_iteration2_hidden.json", "5581ddab5b0a7c191527655c8d74130f598eade52f48c239417b817771fb93fc"),
        ):
            self.assertEqual(sha256((ROOT / "artifacts/iteration2" / name).read_bytes()).hexdigest(), digest)

    def test_official_inspectors_include_complete_license_texts(self):
        texts = [(ROOT / "licenses" / name).read_text() for name in
                 ("jacobian-lens-APACHE-2.0.txt", "d3-ISC.txt")]
        self.assertIn("Apache License", texts[0])
        self.assertIn("Copyright 2010-2023 Mike Bostock", texts[1])
        self.assertIn("Permission to use, copy, modify", texts[1])
        for name in ("qwen_spider_slice.html", "qwen_ascii_face_slice.html"):
            source = (ROOT / "artifacts" / name).read_text()
            self.assertIn("Modified in this repository", source)
            for text in texts:
                self.assertIn(text, source)

    def test_snapshot_and_prompt_provenance_are_explicit(self):
        guide = (ROOT / "understanding.md").read_text()
        method = (ROOT / "iteration2.md").read_text()
        self.assertIn("Iteration 1 snapshot", guide)
        self.assertIn("[iteration2.md](iteration2.md)", guide)
        self.assertIn("The prompts are hand-designed", method)
        self.assertNotIn("new lens fitting, synthetic data", method)
        self.assertIn("accessibility acceptance therefore remain **pending**", method)


if __name__ == "__main__":
    unittest.main()
