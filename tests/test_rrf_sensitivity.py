"""Weighted fusion checks with small rankings, independent of models or APIs."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rrf_sensitivity", ROOT / "benchmarks/public_retrieval/rrf_sensitivity.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from benchmarks.public_retrieval.run import fuse


class WeightedRRFTests(unittest.TestCase):
    def test_author_requested_revised_grid(self):
        self.assertEqual(module.K_VALUES, (10, 20, 40, 60, 80, 90))
        self.assertEqual(len(module.CONFIGS), 30)
        self.assertNotIn(100, module.K_VALUES)

    def test_default_matches_original_with_document_id_ties(self):
        for a, b in [(["b", "a"], ["a", "b"]), (["z", "a"], ["y", "a"]), ([], ["c", "b"])]:
            self.assertEqual(module.weighted_rrf(a, b), fuse(a, b))

    def test_zero_weight_excludes_other_branch_and_endpoints_preserve_order(self):
        for k in module.K_VALUES:
            self.assertEqual(module.weighted_rrf(["a", "c"], ["b", "c"], k, 0), ["b", "c"])
            self.assertEqual(module.weighted_rrf(["a", "c"], ["b", "c"], k, 1), ["a", "c"])
        self.assertEqual(module.weighted_rrf(["a"], [], alpha=0), [])

    def test_weights_can_change_the_winning_document(self):
        self.assertEqual(module.weighted_rrf(["a"], ["b"], alpha=.75)[0], "a")
        self.assertEqual(module.weighted_rrf(["a"], ["b"], alpha=.25)[0], "b")

    def test_smoothing_changes_consensus_versus_single_branch_priority(self):
        lexical = ["single", "consensus"]
        dense = ["x", "y", "consensus"]
        self.assertEqual(module.weighted_rrf(lexical, dense, k=0)[0], "single")
        self.assertEqual(module.weighted_rrf(lexical, dense, k=60)[0], "consensus")

    def test_missing_branch_does_not_get_artificial_duplicate_contribution(self):
        self.assertEqual(module.weighted_rrf([], []), [])
        self.assertEqual(module.weighted_rrf(["b", "a"], []), ["b", "a"])
        self.assertEqual(module.weighted_rrf(["b", "a"], [], cutoff=1), ["b"])

    def test_invalid_candidates_and_parameters_fail(self):
        for args in [(["a", "a"], []), ([], ["a", "a"])]:
            with self.assertRaises(ValueError):
                module.weighted_rrf(*args)
        for kwargs in [{"k": -1}, {"k": float("nan")}, {"alpha": 1.1}, {"alpha": float("nan")}, {"cutoff": -1}]:
            with self.assertRaises(ValueError):
                module.weighted_rrf([], [], **kwargs)

    def test_bootstrap_uses_identical_draws_for_paired_configurations(self):
        values = np.array([[[.1], [.3]], [[.4], [.6]], [[.7], [.9]]])
        boot = module.bootstrap_means(values, repetitions=100, seed=7)
        np.testing.assert_allclose(boot[:, 1] - boot[:, 0], .2, atol=1e-15)
        np.testing.assert_array_equal(boot, module.bootstrap_means(values, repetitions=100, seed=7))


if __name__ == "__main__":
    unittest.main()
