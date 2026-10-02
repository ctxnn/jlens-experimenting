"""Value-level release checks against the unchanged GPU evidence archives."""

import ast
import json
import unittest
from pathlib import Path


class EvidenceFidelityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads(Path("artifacts/qwen_relay_suite.json").read_text())
        cls.compact = json.loads(Path("artifacts/qwen_relay_suite_summary.json").read_text())

    def test_every_projected_rank_and_alignment_matches_archive(self):
        self.assertEqual(len(self.compact["results"]), len(self.raw["results"]))
        for original, projected in zip(self.raw["results"], self.compact["results"]):
            with self.subTest(prompt=original["id"]):
                self.assertEqual(projected["id"], original["id"])
                for key in ("readout_prefix", "readout_token", "content_token_index", "summary"):
                    self.assertEqual(projected[key], original[key])
                self.assertEqual(projected["missing_surfaces"],
                                 [key for key, values in original["resolved_variants"].items() if not values])
                self.assertEqual([point[0] for point in projected["layers"]], list(range(31)))
                for point in projected["layers"]:
                    expected = [point[0]]
                    layer = original["layers"][str(point[0])]
                    for branch in ("jlens", "logit_lens"):
                        for concept in ("target", "answer", "decoy"):
                            best = layer[branch][concept]["best"]
                            expected.append(best["rank"] if best else None)
                    self.assertEqual(point, expected)

    def test_matched_control_uses_the_declared_rule(self):
        counts = {"hidden": 0, "explicit": 0}
        for raw, row in zip(self.raw["results"], self.compact["results"]):
            targets = [(point[4], point[0]) for point in row["layers"] if point[4] is not None]
            rank, layer = min(targets) if targets else (None, None)
            commits = [point[0] for point in row["layers"]
                       if point[5] is not None and point[5] <= 10]
            commit = min(commits) if commits else None
            relay = bool(raw["correct"] and rank is not None and rank <= 100
                         and commit is not None and layer < commit)
            expected = {"target_peak": {"layer": layer, "rank": rank} if rank is not None else None,
                        "answer_commit_layer": commit, "relay": relay}
            self.assertEqual(row["logit_summary"], expected, row["id"])
            counts[row["condition"]] += relay
        self.assertEqual(counts, {"hidden": 6, "explicit": 6})
        self.assertEqual(self.compact["comparison"], {
            "all": {"total": 63, "jlens_relay": 7, "logit_relay": 12},
            "hidden": {"total": 51, "jlens_relay": 7, "logit_relay": 6},
            "explicit": {"total": 12, "jlens_relay": 0, "logit_relay": 6},
        })

    def test_embedded_relay_data_matches_compact_projection(self):
        html = Path("visualizations/qwen-concept-relay.html").read_text()
        encoded = html.split("    const DATA = ", 1)[1].split(";\n    const COL", 1)[0]
        self.assertEqual(json.loads(encoded), self.compact)

    def test_atlas_keeps_complete_unmodified_token_lists_and_prompts(self):
        raw = json.loads(Path("artifacts/qwen_validation.json").read_text())
        html = Path("visualizations/jlens-concept-atlas.html").read_text()
        encoded = html.split("    const prompts = ", 1)[1].split(";\n\n    let activePrompt", 1)[0]
        data = json.loads(encoded)
        notebook = json.loads(Path("notebooks/qwen_official_jlens.ipynb").read_text())
        cell = next(cell for cell in notebook["cells"]
                    if cell["cell_type"] == "code" and "".join(cell["source"]).startswith("PROMPTS ="))
        prompts = ast.literal_eval(ast.parse("".join(cell["source"])).body[0].value)
        self.assertEqual(set(data), set(raw["prompts"]))
        for name, prompt in raw["prompts"].items():
            self.assertEqual(data[name]["text"], prompts[name])
            self.assertEqual(data[name]["model"], prompt["model_top10"])
            self.assertEqual(set(data[name]["layers"]), set(prompt["layers"]))
            for layer, values in prompt["layers"].items():
                self.assertEqual(data[name]["layers"][layer], {
                    "j": values["jlens_top10"], "l": values["logit_lens_top10"],
                    "o": values["top10_overlap"],
                })
                self.assertEqual(len(data[name]["layers"][layer]["j"]), 10)
                self.assertEqual(len(data[name]["layers"][layer]["l"]), 10)


if __name__ == "__main__":
    unittest.main()
