"""
analysis/clustering.py - Spatial Clustering of Conserved Residues

Tests whether conserved residues form a statistically significant spatial
cluster using a permutation test on mean pairwise Cα distances.

Workflow:
  1. Select top 10% most conserved residues (lowest entropy)
  2. Extract Cα coordinates from 3D structure
  3. Compute mean pairwise distance
  4. Permutation test (1000 random draws) → p-value
  5. If p < 0.05 → significant cluster
"""

import math
import random
import logging
from Bio.PDB import PDBParser

logger = logging.getLogger(__name__)

# Three-letter to one-letter mapping for residue identification
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
                if residue.id[0] == " " and residue.get_resname().strip() in THREE_TO_ONE and "CA" in residue
            )
            if count > best_count:
                best_chain = chain
                best_count = count
        return best_chain
    return None


def _residue_label(resseq: int, icode: str) -> str:
    icode = (icode or "").strip()
    return f"{resseq}{icode}" if icode else str(resseq)


def _extract_ca_coords(pdb_path: str) -> tuple[dict, str]:
    """
    Extract Cα coordinates from a PDB file.

    Returns:
        Tuple:
          - Dict mapping (chain_id, resid, icode) -> residue metadata + coordinates.
          - Primary chain id used for extraction.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    coords = {}
    chain = _select_primary_chain(structure)
    if chain is None:
        return coords, ""

    for residue in chain:
        if residue.id[0] != " ":
            continue
        resname = residue.get_resname().strip()
        if resname not in THREE_TO_ONE or "CA" not in residue:
            continue

        resid = residue.id[1]
        icode = residue.id[2].strip()
        key = (chain.id, resid, icode)
        ca = residue["CA"]
        coords[key] = {
            "coord": tuple(ca.get_vector().get_array()),
            "resid": resid,
            "icode": icode,
            "resid_label": _residue_label(resid, icode),
            "aa": THREE_TO_ONE[resname],
            "chain": chain.id,
        }

    return coords, chain.id


def _mean_pairwise_distance(coords_list: list) -> float:
    """Compute mean pairwise Euclidean distance between 3D points."""
    n = len(coords_list)
    if n < 2:
        return 0.0

    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords_list[i][0] - coords_list[j][0]
            dy = coords_list[i][1] - coords_list[j][1]
            dz = coords_list[i][2] - coords_list[j][2]
            total += math.sqrt(dx * dx + dy * dy + dz * dz)
            count += 1

    return total / count if count > 0 else 0.0


def cluster_conserved(
    pdb_path: str,
    entropy_scores: dict,
    top_percent: float = 0.1,
    n_permutations: int = 1000,
) -> dict:
    """
    Test whether conserved residues form a significant spatial cluster.

    Args:
        pdb_path: Path to PDB file.
        entropy_scores: Dict {resid_1based: entropy_score}.
        top_percent: Fraction of most conserved residues to select (default 0.1).
        n_permutations: Number of random permutations for p-value (default 1000).

    Returns:
        Dictionary with:
          - 'residues': list of {resid, aa, entropy} for conserved cluster
          - 'num_residues': int
          - 'mean_distance': float (Å)
          - 'p_value': float
          - 'significant': bool (p < 0.05)
          - 'message': human-readable summary
    """
    logger.info("Running spatial clustering analysis (top %.0f%%, %d permutations)",
                top_percent * 100, n_permutations)

    # Extract Cα coordinates
    try:
        all_coords, primary_chain = _extract_ca_coords(pdb_path)
    except Exception as e:
        logger.error("Failed to extract Cα coordinates: %s", e)
        return {
            "residues": [], "num_residues": 0,
            "mean_distance": 0, "p_value": 1.0,
            "significant": False, "message": f"Error: {e}",
        }

    if len(all_coords) < 10:
        return {
            "residues": [], "num_residues": 0,
            "mean_distance": 0, "p_value": 1.0,
            "significant": False,
            "message": "Too few residues with coordinates for clustering analysis",
        }

    # Select top conserved residues that have coordinates
    scored_residues = []
    for resid, score in entropy_scores.items():
        matches = [
            key for key, info in all_coords.items()
            if info["resid"] == resid
        ]
        for key in matches:
            scored_residues.append((key, score))
    scored_residues.sort(key=lambda x: x[1])  # lowest entropy = most conserved

    n_select = max(3, int(len(scored_residues) * top_percent))
    conserved = scored_residues[:n_select]
    conserved_keys = [r[0] for r in conserved]

    if len(conserved_keys) < 3:
        return {
            "residues": [], "num_residues": 0,
            "mean_distance": 0, "p_value": 1.0,
            "significant": False,
            "message": "Too few conserved residues with coordinates",
        }

    # Compute observed mean pairwise distance
    conserved_coords = [all_coords[k]["coord"] for k in conserved_keys]
    observed_dist = _mean_pairwise_distance(conserved_coords)

    # Permutation test
    all_resids_with_coords = list(all_coords.keys())
    count_le = 0
    random.seed(42)  # reproducibility

    for _ in range(n_permutations):
        random_resids = random.sample(all_resids_with_coords, len(conserved_keys))
        random_coords = [all_coords[r]["coord"] for r in random_resids]
        random_dist = _mean_pairwise_distance(random_coords)
        if random_dist <= observed_dist:
            count_le += 1

    p_value = round(count_le / n_permutations, 4)
    significant = p_value < 0.05

    cluster_residues = []
    for key, entropy in conserved:
        info = all_coords[key]
        cluster_residues.append({
            "resid": info["resid"],
            "resid_label": info["resid_label"],
            "chain": info["chain"],
            "icode": info["icode"],
            "aa": info["aa"],
            "entropy": entropy,
        })

    if significant:
        message = (
            f"Conserved residues form a statistically significant spatial cluster "
            f"(p = {p_value:.4f}). Mean pairwise distance: {observed_dist:.1f} Å "
            f"across {len(conserved_keys)} residues on chain {primary_chain}."
        )
    else:
        message = (
            f"No significant spatial clustering of conserved residues detected "
            f"(p = {p_value:.4f}). Mean pairwise distance: {observed_dist:.1f} Å."
        )

    logger.info("Clustering: mean_dist=%.1f Å, p=%.4f, significant=%s",
                observed_dist, p_value, significant)

    return {
        "residues": cluster_residues,
        "num_residues": len(conserved_keys),
        "mean_distance": round(observed_dist, 2),
        "p_value": p_value,
        "significant": significant,
        "message": message,
    }
