import unittest

from reasoning.inference_engine import infer_function


class InferenceTest(unittest.TestCase):
    def test_protease_annotation_without_motif_is_putative(self):
        data = {
            "sequence": "ACDEFGHIKLMNPQRSTVWY",
            "tm_prediction": {"tm_helices": 0, "signal_peptide": False},
            "blast": {"hits": [{"definition": "membrane protease family protein"}]},
            "domains": [],
            "conservation": {"conserved_positions": [], "confidence": "Medium"},
            "structure": {},
            "ligands": {"ligands": []},
        }
        result = infer_function(data)
        self.assertIn("Putative protease", result["function"])
        self.assertNotIn("Serine protease", result["function"])

    def test_low_conservation_caps_high_confidence(self):
        data = {
            "sequence": "ACDEFGHIKLMNPQRSTVWY",
            "tm_prediction": {"tm_helices": 1, "signal_peptide": False},
            "blast": {"hits": [{"definition": "receptor protein kinase"}]},
            "domains": [{"name": "kinase"}],
            "conservation": {"conserved_positions": [], "confidence": "Low"},
            "structure": {},
            "ligands": {"ligands": [{"name": "ATP"}]},
        }
        result = infer_function(data)
        self.assertEqual(result["confidence"], "Medium")
        self.assertTrue(any("Guardrail" in r for r in result["rules_triggered"]))


if __name__ == "__main__":
    unittest.main()
