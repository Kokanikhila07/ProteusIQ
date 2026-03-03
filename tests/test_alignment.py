import unittest
from unittest.mock import patch

from tools import alignment


class AlignmentTest(unittest.TestCase):
    def _blast_hits(self, n):
        return [{"accession": f"P{10000+i}", "hit_id": "", "definition": ""} for i in range(n)]

    def test_conservation_requires_five_unique_homologs(self):
        query = "ACDEFGHIKL"
        hits = self._blast_hits(6)
        seq_map = {
            "P10000": "ACDEFGHIKL",
            "P10001": "ACDEYGHIKL",
            "P10002": "ACDEFGHIKM",
            "P10003": "ACDEFGHIKN",
            "P10004": "ACDEFGHIKQ",
            "P10005": "ACDEFGHIKR",
        }

        def fake_fetch(accession):
            return seq_map[accession]

        def fake_align(query_seq, subject_seq):
            return {i: aa for i, aa in enumerate(subject_seq)}

        with patch("tools.alignment._fetch_sequence", side_effect=fake_fetch), patch(
            "tools.alignment._align_sequences", side_effect=fake_align
        ), patch("tools.alignment.time.sleep", return_value=None):
            result = alignment.compute_conservation(query, hits)

        self.assertFalse(result["skipped"])
        self.assertEqual(result["num_sequences"], 5)  # one duplicate equals query, dropped
        self.assertEqual(result["confidence"], "Medium")
        self.assertIn("limited_homolog_depth", result["quality_flags"])

    def test_conservation_skips_below_threshold(self):
        query = "ACDEFGHIKL"
        hits = self._blast_hits(4)
        result = alignment.compute_conservation(query, hits)
        self.assertTrue(result["skipped"])
        self.assertEqual(result["confidence"], "Low")


if __name__ == "__main__":
    unittest.main()
