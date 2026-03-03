import os
import tempfile
import unittest

from analysis.contacts import quantify_contacts


PDB_TEXT = """\
ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  ALA A   1      11.500  10.000  10.000  1.00 20.00           C
ATOM      3  CB  ALA A   1      13.900  10.000  10.000  1.00 20.00           C
ATOM      4  N   GLY A   2      20.000  20.000  20.000  1.00 20.00           N
ATOM      5  CA  GLY A   2      21.000  20.000  20.000  1.00 20.00           C
HETATM    6  C1  LIG Z 501      15.000  10.000  10.000  1.00 20.00           C
END
"""


class ContactsTest(unittest.TestCase):
    def test_heavy_atom_contacts_detected_when_ca_is_far(self):
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as handle:
            handle.write(PDB_TEXT)
            pdb_path = handle.name

        try:
            result = quantify_contacts(
                pdb_path,
                entropy_scores={1: 0.1},
                conservation_classes={1: "highly_conserved"},
                distance_cutoff=4.0,
            )
            self.assertEqual(result["total_ligands"], 1)
            lig = result["ligand_contacts"][0]
            self.assertEqual(lig["num_contacts"], 1)
            contact = lig["contacts"][0]
            self.assertEqual(contact["resid"], 1)
            self.assertEqual(contact["chain"], "A")
            self.assertAlmostEqual(contact["distance"], 1.1, places=1)
        finally:
            os.unlink(pdb_path)


if __name__ == "__main__":
    unittest.main()
