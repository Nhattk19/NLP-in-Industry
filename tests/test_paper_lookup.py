"""Regression tests for references to papers absent from the abstract index."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock

spec = importlib.util.spec_from_file_location(
    "paper_lookup", Path(__file__).resolve().parents[1] / "src/web/core/paper_lookup.py")
lookup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lookup)


class PaperLookupTests(unittest.TestCase):
    def test_abstract_match_does_not_read_fulltext(self):
        collection = Mock()
        paper = {"paper_id": "known", "title": "Known paper"}
        self.assertEqual(lookup.resolve_detail_paper("known", [], [paper], collection), paper)
        collection.get.assert_not_called()

    def test_fulltext_only_reference(self):
        collection = Mock()
        collection.get.return_value = {"metadatas": [{
            "paper_id": "arxiv:123", "title": "External paper",
            "source_url": "https://arxiv.org/abs/123"}]}
        paper = lookup.resolve_detail_paper("arxiv:123", [], [], collection)
        self.assertEqual(paper["paper_id"], "arxiv:123")
        self.assertEqual(paper["arxiv"], "https://arxiv.org/abs/123")
        collection.get.assert_called_once_with(
            where={"paper_id": "arxiv:123"}, limit=1, include=["metadatas"])

    def test_title_fallback(self):
        collection = Mock()
        collection.get.side_effect = [{"metadatas": []}, {"metadatas": [
            {"paper_id": "123", "title": "A paper"}]}]
        self.assertEqual(lookup.resolve_detail_paper("A paper", [], [], collection)["paper_id"], "123")

    def test_missing_reference(self):
        collection = Mock()
        collection.get.return_value = {"metadatas": []}
        self.assertIsNone(lookup.resolve_detail_paper("missing", [], [], collection))


if __name__ == "__main__":
    unittest.main()
