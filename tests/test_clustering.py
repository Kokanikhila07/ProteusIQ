import os
import tempfile
import unittest

from analysis.clustering import cluster_conserved


def _make_pdb() -> str:
    lines = []
    atom_id = 1
    for resid in range(1, 13):
        x = float(resid)
        lines.append(
            f"ATOM  {atom_id:5d}  CA  ALA A{resid:4d}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C"
        )
        atom_id += 1
    for resid in range(1, 3):
        x = float(100 + resid)
        lines.append(
            f"ATOM  {atom_id:5d}  CA  GLY B{resid:4d}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C"
        )
        atom_id += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


class ClusteringTest(unittest.TestCase):
    def test_cluster_uses_single_primary_chain(self):
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as handle:
            handle.write(_make_pdb())
            pdb_path = handle.name

        try:
            entropy_scores = {i: i / 100.0 for i in range(1, 11)}
            result = cluster_conserved(pdb_path, entropy_scores, top_percent=0.3, n_permutations=50)
            self.assertGreaterEqual(result["num_residues"], 3)
            self.assertTrue(all(r["chain"] == "A" for r in result["residues"]))
        finally:
            os.unlink(pdb_path)


if __name__ == "__main__":
    unittest.main()
