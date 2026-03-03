"""
analysis/secondary_structure.py – Secondary Structure from PDB Headers

Parses HELIX and SHEET records from PDB files to extract experimentally
determined secondary structure assignments. No heuristic predictions —
only experimental data from PDB headers.

Output format: per-residue assignment as H (helix), E (sheet), C (coil).
"""

import logging
from Bio.PDB import PDBParser

logger = logging.getLogger(__name__)


def parse_secondary_structure(pdb_path: str, chain_id: str = None) -> dict:
    """
    Parse secondary structure from PDB HELIX/SHEET records.

    Args:
        pdb_path: Path to PDB file.
        chain_id: Optional chain to filter. If None, uses first chain found.

    Returns:
        Dictionary with:
          - 'assignments': dict {resid_1based: 'H'|'E'|'C'}
          - 'helix_ranges': list of (start, end) tuples
          - 'sheet_ranges': list of (start, end) tuples
          - 'helix_pct': float
          - 'sheet_pct': float
          - 'coil_pct': float
          - 'summary': human-readable summary
          - 'ss_string': string of H/E/C characters (for overlay on sequence)
    """
    logger.info("Parsing secondary structure from PDB: %s", pdb_path)

    helix_ranges = []
    sheet_ranges = []

    try:
        # Read raw PDB lines for HELIX/SHEET records
        with open(pdb_path, "r") as f:
            for line in f:
                record = line[:6].strip()

                if record == "HELIX":
                    try:
                        init_chain = line[19].strip()
                        init_seq = int(line[21:25].strip())
                        end_chain = line[31].strip()
                        end_seq = int(line[33:37].strip())

                        if chain_id is None or init_chain == chain_id:
                            helix_ranges.append((init_seq, end_seq))
                            if chain_id is None:
                                chain_id = init_chain
                    except (ValueError, IndexError):
                        continue

                elif record == "SHEET":
                    try:
                        init_chain = line[21].strip()
                        init_seq = int(line[22:26].strip())
                        end_chain = line[32].strip()
                        end_seq = int(line[33:37].strip())

                        if chain_id is None or init_chain == chain_id:
                            sheet_ranges.append((init_seq, end_seq))
                            if chain_id is None:
                                chain_id = init_chain
                    except (ValueError, IndexError):
                        continue

    except Exception as e:
        logger.error("Failed to read PDB file for SS: %s", e)
        return {
            "assignments": {}, "helix_ranges": [], "sheet_ranges": [],
            "helix_pct": 0, "sheet_pct": 0, "coil_pct": 0,
            "summary": f"Error: {e}", "ss_string": "",
        }

    # Determine residue range from Bio.PDB
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        residue_ids = []
        for model in structure:
            for chain in model:
                if chain_id and chain.id != chain_id:
                    continue
                for residue in chain:
                    if residue.id[0] == " ":  # standard AA
                        residue_ids.append(residue.id[1])
                if residue_ids:
                    break
            break
    except Exception:
        residue_ids = []

    if not residue_ids:
        return {
            "assignments": {}, "helix_ranges": helix_ranges,
            "sheet_ranges": sheet_ranges,
            "helix_pct": 0, "sheet_pct": 0, "coil_pct": 0,
            "summary": "No standard residues found in structure",
            "ss_string": "",
        }

    # Build per-residue assignment
    assignments = {}
    for resid in residue_ids:
        assignments[resid] = "C"  # default to coil

    for start, end in helix_ranges:
        for resid in range(start, end + 1):
            if resid in assignments:
                assignments[resid] = "H"

    for start, end in sheet_ranges:
        for resid in range(start, end + 1):
            if resid in assignments:
                assignments[resid] = "E"

    # Statistics
    total = len(assignments)
    n_helix = sum(1 for v in assignments.values() if v == "H")
    n_sheet = sum(1 for v in assignments.values() if v == "E")
    n_coil = total - n_helix - n_sheet

    helix_pct = round(n_helix / total * 100, 1) if total > 0 else 0
    sheet_pct = round(n_sheet / total * 100, 1) if total > 0 else 0
    coil_pct = round(n_coil / total * 100, 1) if total > 0 else 0

    # Build SS string in residue order
    sorted_resids = sorted(assignments.keys())
    ss_string = "".join(assignments[r] for r in sorted_resids)

    summary = f"Helix: {helix_pct}%, Sheet: {sheet_pct}%, Coil: {coil_pct}%"

    logger.info("Secondary structure: %s (%d residues)", summary, total)

    return {
        "assignments": assignments,
        "helix_ranges": helix_ranges,
        "sheet_ranges": sheet_ranges,
        "helix_pct": helix_pct,
        "sheet_pct": sheet_pct,
        "coil_pct": coil_pct,
        "summary": summary,
        "ss_string": ss_string,
    }
