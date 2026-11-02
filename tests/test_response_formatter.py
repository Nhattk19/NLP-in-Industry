"""Paper references must stay unique and agree with the answer labels."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "response_formatter",
    Path(__file__).resolve().parents[1] / "src/agent/nodes/response_formatter.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ResponseFormatterTests(unittest.TestCase):
    def format(self, answer, docs, refs):
        return module.ResponseFormatter()({
            "initial_answer": answer,
            "context_documents": docs,
            "answer_citations": [{"ref_number": n} for n in refs],
        })

    def test_multiple_chunks_of_one_paper(self):
        docs = [
            {"paper_id": "a", "chunk_id": "a0", "title": "Argument Mining"},
            {"paper_id": "s", "chunk_id": "s0", "title": "SuperGLEBer"},
            {"paper_id": "s", "chunk_id": "s1", "title": "SuperGLEBer"},
            {"paper_id": "s", "chunk_id": "s2", "title": "SuperGLEBer"},
        ]
        result = self.format("First [2]. Second [1][3][4].", docs, [2, 1, 3, 4])
        self.assertEqual(result["final_answer"], "First [1]. Second [2][1].")
        self.assertEqual([s["paper_id"] for s in result["final_sources"]], ["s", "a"])
        self.assertEqual([s["ref_number"] for s in result["final_sources"]], [1, 2])
        self.assertEqual(result["initial_answer"], "First [2]. Second [1][3][4].")

    def test_sparse_chunk_labels_become_consecutive(self):
        docs = [{"paper_id": str(n)} for n in range(5)]
        result = self.format("A [1]. B [2]. C [5].", docs, [1, 2, 5])
        self.assertEqual(result["final_answer"], "A [1]. B [2]. C [3].")
        self.assertEqual([s["ref_number"] for s in result["final_sources"]], [1, 2, 3])

    def test_missing_ids_use_normalized_title(self):
        docs = [
            {"paper_id": "N/A", "title": "  Same  Paper "},
            {"title": "same paper"},
            {"title": "Other paper"},
        ]
        result = self.format("Claim [1][2][3].", docs, [1, 2, 3])
        self.assertEqual(result["final_answer"], "Claim [1][2].")
        self.assertEqual(len(result["final_sources"]), 2)

    def test_uncited_fallback_groups_papers_without_using_rank(self):
        docs = [{"paper_id": "a", "rank": 5}, {"paper_id": "a", "rank": 8},
                {"paper_id": "b", "rank": 10}]
        result = self.format("No citations.", docs, [])
        self.assertEqual([s["ref_number"] for s in result["final_sources"]], [1, 2])
        self.assertEqual(result["final_answer"], "No citations.")

    def test_distinct_paper_ids_with_same_title_are_preserved(self):
        result = self.format("A [1][2].", [
            {"paper_id": "a", "title": "Shared title"},
            {"paper_id": "b", "title": "Shared title"},
        ], [1, 2])
        self.assertEqual(len(result["final_sources"]), 2)


if __name__ == "__main__":
    unittest.main()
