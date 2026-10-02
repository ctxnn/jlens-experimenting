"""CPU checks for the exact coordinate-edit math used in the Colab runner."""
import importlib.util
import json
import re
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("iteration2", ROOT / "experiments/qwen_iteration2.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CoordinateSwapTests(unittest.TestCase):
    def setUp(self):
        self.v = np.array([[2, 1], [0, 3], [1, 2], [4, -1]], dtype=np.float32)
        self.h = np.array([[3, -2, 5, 1], [0, 7, 2, -3]], dtype=np.float32)

    def test_exchanges_coordinates(self):
        before = self.h @ np.linalg.pinv(self.v).T
        after = self.h + MODULE.coordinate_swap_delta(self.h, self.v)
        np.testing.assert_allclose(after @ np.linalg.pinv(self.v).T, before[:, ::-1], atol=2e-6)

    def test_preserves_orthogonal_complement(self):
        projection = self.v @ np.linalg.pinv(self.v)
        edited = self.h + MODULE.coordinate_swap_delta(self.h, self.v)
        np.testing.assert_allclose(edited @ (np.eye(4) - projection),
                                   self.h @ (np.eye(4) - projection), atol=2e-6)

    def test_is_involution(self):
        edited = self.h + MODULE.coordinate_swap_delta(self.h, self.v)
        twice = edited + MODULE.coordinate_swap_delta(edited, self.v)
        np.testing.assert_allclose(twice, self.h, atol=2e-6)

    def test_zero_strength_is_no_op(self):
        np.testing.assert_array_equal(MODULE.coordinate_swap_delta(self.h, self.v, 0), 0)

    def test_rejects_dependent_vectors(self):
        with self.assertRaises(ValueError):
            MODULE.coordinate_swap_delta(self.h, np.ones((4, 2)))

    def test_rejects_wrong_shape_and_nonfinite(self):
        with self.assertRaises(ValueError):
            MODULE.coordinate_swap_delta(self.h, np.ones((3, 2)))
        with self.assertRaises(ValueError):
            MODULE.coordinate_swap_delta(self.h * np.nan, self.v)

    def test_random_control_norm_and_reproducibility(self):
        delta = MODULE.coordinate_swap_delta(self.h, self.v)
        noise = MODULE.norm_matched_delta(delta, 7)
        np.testing.assert_allclose(np.linalg.norm(noise, axis=-1), np.linalg.norm(delta, axis=-1), rtol=1e-6)
        np.testing.assert_array_equal(noise, MODULE.norm_matched_delta(delta, 7))
        self.assertFalse(np.allclose(noise, delta))

    def test_frozen_protocol(self):
        protocol = json.loads((ROOT / "experiments/iteration2_prompts.json").read_text())
        self.assertEqual(len(protocol["cases"]), 20)
        self.assertEqual(len({c["id"] for c in protocol["cases"]}), 20)
        self.assertEqual(protocol["readout_suffix"], " ")
        self.assertIn(protocol["generation_layer"], protocol["intervention_layers"])
        self.assertEqual(protocol["cases"][0]["target"], "spider")

    def test_hidden_followup_has_named_pairs_but_no_target_words(self):
        hidden = json.loads((ROOT / "experiments/iteration2_hidden_prompts.json").read_text())
        explicit = json.loads((ROOT / "experiments/iteration2_prompts.json").read_text())
        by_id = {c["id"]: c for c in explicit["cases"]}
        self.assertEqual(len(hidden["cases"]), 20)
        for c in hidden["cases"]:
            self.assertIsNone(re.search(r"\b" + re.escape(c["target"]) + r"\b", c["prompt"], re.I), c["id"])
            original = by_id[c["paired_explicit_id"]]
            for key in ("target", "decoy", "answer", "alternative", "answer_aliases", "alternative_aliases"):
                self.assertEqual(c[key], original[key])


if __name__ == "__main__":
    unittest.main()
