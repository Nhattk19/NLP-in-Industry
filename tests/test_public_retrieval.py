"""Check scoring against small hand-computable relevance/ranking examples."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('public_retrieval',Path(__file__).resolve().parents[1]/'benchmarks/public_retrieval/run.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PublicRetrievalTests(unittest.TestCase):
    def test_binary_known_targets(self):
        scores=module.metrics(['unjudged','a'],{'a':1,'b':1},10)
        self.assertEqual(scores['MRR@10'],.5)
        self.assertEqual(scores['Recall@10'],.5)
        self.assertEqual(scores['Hit@10'],1.)

    def test_graded_perfect_order_and_zero_grade(self):
        scores=module.metrics(['a','b','c'],{'a':3,'b':2,'c':0},10)
        self.assertEqual(scores['nDCG@10'],1.)
        self.assertEqual(scores['Recall@10'],1.)

    def test_zero_relevant_and_cutoff(self):
        self.assertEqual(module.metrics(['x','a'],{'a':1},1)['Hit@1'],0.)
        self.assertTrue(all(v==0 for v in module.metrics(['x'],{'x':0},10).values()))

    def test_supports_binary_relevance_threshold(self):
        self.assertEqual(module.metrics(['a'],{'a':1,'b':2},10,threshold=2)['Hit@10'],0.)

    def test_rrf_overlap_and_empty_branch(self):
        self.assertEqual(module.fuse(['a','b'],['b','c'])[0],'b')
        self.assertEqual(module.fuse(['a','b'],[]),['a','b'])

    def test_rrf_rejects_duplicate_candidates(self):
        with self.assertRaises(AssertionError):module.fuse(['a','a'],[])


if __name__=='__main__':unittest.main()
