"""Independent value-level checks for the saved Iteration 2 figures and archives."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "artifacts/iteration2/qwen_iteration2.json"
HIDDEN = ROOT / "artifacts/iteration2/qwen_iteration2_hidden.json"


@unittest.skipUnless(RAW.exists(), "GPU archive has not been downloaded")
class Iteration2EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports = [json.loads(RAW.read_text())]
        if HIDDEN.exists():
            cls.reports.append(json.loads(HIDDEN.read_text()))
        cls.page = (ROOT / "visualizations/qwen-iteration2.html").read_text()
        cls.embedded = json.loads(re.search(r'<script type="application/json" id="dataset">(.*?)</script>', cls.page, re.S)[1])

    def test_all_planned_cases_are_archived(self):
        for report in self.reports:
            self.assertEqual([c["id"] for c in report["cases"]], [c["id"] for c in report["protocol"]["cases"]])
            self.assertEqual(len(report["cases"]), 20)
            for case in report["cases"]:
                self.assertEqual([r["layer"] for r in case["readouts"]], list(range(31)))
                self.assertEqual(len(case["interventions"]), 30)

    def test_embedding_preserves_each_raw_measurement(self):
        indexed = {c["id"]: c for c in self.embedded["cases"]}
        self.assertEqual(len(indexed), sum(len(r["cases"]) for r in self.reports))
        for report in self.reports:
            for case in report["cases"]:
                actual = dict(indexed[case["id"]])
                actual.pop("run_kind")
                self.assertEqual(actual, case)

    def test_source_hashes_and_pilot_are_preserved(self):
        paths = {RAW.name: RAW, HIDDEN.name: HIDDEN}
        for source in self.embedded["source_archives"]:
            path = paths[Path(source["path"]).name]
            self.assertEqual(source["sha256"], sha256(path.read_bytes()).hexdigest())
        pilot = json.loads((ROOT / "artifacts/iteration2/qwen_iteration2.pilot.json").read_text())
        self.assertEqual(pilot["cases"], self.reports[0]["cases"][:3])

    def test_no_op_and_parity_are_real(self):
        for report in self.reports:
            for case in report["cases"]:
                self.assertLessEqual(case["final_readout_max_abs_logit_error"], .25)
                for row in case["interventions"]:
                    if row["condition"] == "no_op" and row["available"]:
                        self.assertEqual(row["max_abs_logit_change"], 0)
                        self.assertEqual(row["measurements"], case["clean"]["measurements"])
                        self.assertEqual(row["margin_change"], 0 if case["clean"]["alternative_minus_original_mass"] is not None else None)

    def test_derived_margins(self):
        for report in self.reports:
            for case in report["cases"]:
                for row in [case["clean"]] + [r for r in case["interventions"] if r["available"]]:
                    a, b = row["measurements"]["answer"]["mass"], row["measurements"]["alternative"]["mass"]
                    expected = b - a if a is not None and b is not None else None
                    self.assertEqual(expected, row["alternative_minus_original_mass"])
                    if row is not case["clean"] and expected is not None:
                        self.assertAlmostEqual(row["margin_change"], expected - case["clean"]["alternative_minus_original_mass"])

    def test_static_bar_widths_are_raw_probabilities(self):
        ns = {"s": "http://www.w3.org/2000/svg"}
        for report in self.reports:
            for case in report["cases"]:
                svg = ET.parse(ROOT / f'visualizations/iteration2/{case["id"]}-intervention.svg')
                widths = [float(r.attrib["width"]) for r in svg.findall(".//s:rect", ns)
                          if r.attrib.get("fill") in ("#f7ce68", "#dc99ea")]
                expected = []
                for condition in ("clean", "concept_swap", "no_op", "unrelated_swap", "answer_swap", "random_norm_matched"):
                    row = case["clean"] if condition == "clean" else next((r for r in case["interventions"]
                        if r["condition"] == condition and r["layer"] == 24 and r["available"]), None)
                    if row is not None:
                        for key in ("answer", "alternative"):
                            mass = row["measurements"][key]["mass"]
                            if mass is not None:
                                expected.append(round(500 * mass, 3))
                self.assertEqual(widths, expected)

    def test_manifest_matches_generated_viewer(self):
        manifest = json.loads((ROOT / "artifacts/iteration2/run_manifest.json").read_text())
        self.assertEqual(manifest["html_sha256"], sha256(self.page.encode()).hexdigest())
        self.assertEqual(manifest["svg_figures"], 2 * len(self.embedded["cases"]))


if __name__ == "__main__":
    unittest.main()
