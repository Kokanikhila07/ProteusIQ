"""
analysis/contacts.py - Ligand Contact Quantification

Computes quantitative contact details between ligands and protein residues
using atom-level distance calculations. For each ligand, identifies residues
within a heavy-atom distance cutoff, annotates conservation class and
residue chemistry.
"""

import logging
from Bio.PDB import PDBParser

logger = logging.getLogger(__name__)

# Potentially catalytic residue types
CATALYTIC_RESIDUES = frozenset("KRHDECSYT")

# Three-letter to one-letter amino acid mapping
THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def _select_primary_chain(structure):
    """Select the first-model protein chain with the most standard residues."""
    for model in structure:
        best_chain = None
        best_count = -1
        for chain in model:
            count = sum(
                1 for residue in chain
                if residue.id[0] == " " and residue.get_resname().strip() in THREE_TO_ONE
            )
            if count > best_count:
                best_chain = chain
                best_count = count
        return best_chain
    return None


def _heavy_atoms(residue):
    """Return non-hydrogen atoms from a residue."""
    atoms = []
    for atom in residue.get_atoms():
        element = (atom.element or "").strip().upper()
        if element == "H" or atom.get_name().startswith("H"):
            continue
        atoms.append(atom)
    return atoms


def _residue_label(resseq: int, icode: str) -> str:
    """Create human-readable residue label (e.g., 123A for insertion code A)."""
    icode = (icode or "").strip()
    return f"{resseq}{icode}" if icode else str(resseq)


def _get_protein_residues(structure):
    """Extract residues for the primary protein chain with chain-safe keys."""
    chain = _select_primary_chain(structure)
    if chain is None:
        return {}

    residues = {}
    for residue in chain:
        if residue.id[0] != " ":
            continue
        resname = residue.get_resname().strip()
        if resname not in THREE_TO_ONE:
            continue

        resseq = residue.id[1]
        icode = residue.id[2].strip()
        key = (chain.id, resseq, icode)
        atoms = _heavy_atoms(residue)
        if not atoms:
            continue

        residues[key] = {
            "atoms": atoms,
            "aa": THREE_TO_ONE[resname],
            "chain": chain.id,
            "resid": resseq,
            "icode": icode,
            "resid_label": _residue_label(resseq, icode),
        }
    return residues


def _get_ligand_atoms(structure):
    """Extract heavy atoms from HETATM ligand residues (non-water)."""
    ligands = {}
    water_names = {"HOH", "WAT", "H2O"}
    for model in structure:
        for chain in model:
            for residue in chain:
                het_flag = residue.id[0]
                resname = residue.get_resname().strip()
                if het_flag != " " and resname not in water_names:
                    key = (resname, chain.id, residue.id[1], residue.id[2].strip())
                    atoms = _heavy_atoms(residue)
                    if atoms:
                        ligands[key] = atoms
        break  # first model only
    return ligands


def quantify_contacts(
    pdb_path: str,
    entropy_scores: dict = None,
    conservation_classes: dict = None,
    distance_cutoff: float = 4.0,
) -> dict:
    """
    Quantify ligand-residue contacts from a PDB file.

    Args:
        pdb_path: Path to a PDB file.
        entropy_scores: Optional dict {resid_1based: entropy_score}.
        conservation_classes: Optional dict {resid_1based: classification}.
        distance_cutoff: Heavy-atom cutoff in angstroms for contacts (default 4.0).

    Returns:
        Dictionary with per-ligand contact data:
        {
            'ligand_contacts': [
                {
                    'ligand_name': 'ATP',
                    'chain': 'A',
                    'num_contacts': 12,
                    'num_conserved': 8,
                    'num_catalytic': 5,
                    'contacts': [
                        {'resid': 123, 'aa': 'K', 'conservation': 0.15,
                         'conservation_class': 'highly_conserved',
                         'is_catalytic': True, 'distance': 3.2},
                        ...
                    ],
                    'summary': 'ATP contacts 12 residues; 8 highly conserved, 5 catalytic'
                },
                ...
            ],
            'total_ligands': 3,
            'message': 'Ligand contact analysis completed'
        }
    """
    if entropy_scores is None:
        entropy_scores = {}
    if conservation_classes is None:
        conservation_classes = {}

    logger.info("Computing ligand contacts from %s (cutoff %.1f Å)", pdb_path, distance_cutoff)

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
    except Exception as e:
        logger.error("Failed to parse PDB for contacts: %s", e)
        return {
            "ligand_contacts": [],
            "total_ligands": 0,
            "message": f"PDB parse error: {e}",
        }

    protein_residues = _get_protein_residues(structure)
    ligand_atoms = _get_ligand_atoms(structure)

    if not protein_residues:
        return {
            "ligand_contacts": [],
            "total_ligands": 0,
            "message": "No standard protein residues found in structure",
        }

    results = []

    for (lig_name, lig_chain, lig_resid, lig_icode), lig_atom_list in ligand_atoms.items():
        contacts = []
        seen_keys = set()

        for residue_key, residue_info in protein_residues.items():
            if residue_key in seen_keys:
                continue

            min_dist = float("inf")
            min_pair = ("", "")
            protein_atoms = residue_info["atoms"]

            # Find minimum heavy-atom distance between ligand and residue
            for lig_atom in lig_atom_list:
                for prot_atom in protein_atoms:
                    try:
                        d = prot_atom - lig_atom  # Bio.PDB distance operator
                        if d < min_dist:
                            min_dist = d
                            min_pair = (prot_atom.get_name(), lig_atom.get_name())
                    except Exception:
                        continue

            if min_dist <= distance_cutoff:
                seen_keys.add(residue_key)
                resid = residue_info["resid"]
                aa = residue_info["aa"]
                ent = entropy_scores.get(resid, None)
                cls = conservation_classes.get(resid, "unknown")
                is_cat = aa in CATALYTIC_RESIDUES

                contacts.append({
                    "resid": resid,
                    "resid_label": residue_info["resid_label"],
                    "icode": residue_info["icode"],
                    "aa": aa,
                    "chain": residue_info["chain"],
                    "conservation": ent,
                    "conservation_class": cls,
                    "is_catalytic": is_cat,
                    "distance": round(min_dist, 2),
                    "protein_atom": min_pair[0],
                    "ligand_atom": min_pair[1],
                })

        # Sort contacts by distance
        contacts.sort(key=lambda c: c["distance"])

        num_conserved = sum(1 for c in contacts if c["conservation_class"] == "highly_conserved")
        num_catalytic = sum(1 for c in contacts if c["is_catalytic"])
        n = len(contacts)

        summary = f"{lig_name} contacts {n} residue{'s' if n != 1 else ''}"
        if num_conserved > 0 or num_catalytic > 0:
            summary += f"; {num_conserved} highly conserved, {num_catalytic} catalytic"

        results.append({
            "ligand_name": lig_name,
            "chain": lig_chain,
            "ligand_resid": lig_resid,
            "ligand_icode": lig_icode,
            "ligand_resid_label": _residue_label(lig_resid, lig_icode),
            "num_contacts": n,
            "num_conserved": num_conserved,
            "num_catalytic": num_catalytic,
            "contacts": contacts,
            "summary": summary,
        })

    logger.info("Ligand contact analysis: %d ligands analyzed", len(results))

    return {
        "ligand_contacts": results,
        "total_ligands": len(results),
        "message": f"Analyzed contacts for {len(results)} ligand(s)",
    }
