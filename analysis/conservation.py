"""
analysis/conservation.py – Shannon Entropy Conservation Scoring

Computes per-position Shannon entropy from pairwise alignment data to
quantify evolutionary conservation. This replaces the simple fraction-of-
identity approach with the standard measure used in evolutionary analysis.

Entropy:   H = -Σ p_i * log2(p_i)  (over 20 amino acids)
Normalized:  H_norm = H / log2(20)  → range [0, 1]

Classification:
  Highly conserved:     H_norm < 0.2
  Moderately conserved: 0.2 ≤ H_norm < 0.5
  Variable:             H_norm ≥ 0.5
"""

import math
import logging

logger = logging.getLogger(__name__)

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
LOG2_20 = math.log2(20)


def compute_entropy(position_aas: list[str]) -> float:
    """
    Compute normalized Shannon entropy for a single alignment column.

    Uses sample-size-adaptive Laplace smoothing: pseudocount is applied
    only to **observed** amino acid types to avoid inflating entropy
    in small samples (e.g., 2-5 sequences).

    Args:
        position_aas: List of amino acid characters at one position
                      across all aligned sequences.

    Returns:
        Normalized entropy in [0, 1]. 0 = perfectly conserved, 1 = maximally diverse.
    """
    n = len(position_aas)
    if n == 0:
        return 1.0

    # Count frequencies (only valid amino acids)
    counts = {}
    for aa in position_aas:
        aa_upper = aa.upper()
        if aa_upper in AMINO_ACIDS:
            counts[aa_upper] = counts.get(aa_upper, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return 1.0

    # If only one unique amino acid, it's perfectly conserved
    if len(counts) == 1:
        return 0.0

    # Sample-size-adaptive Laplace smoothing
    # Pseudocount scales inversely with sample size so small samples
    # aren't overwhelmed, but fully conserved columns still return ~0
    num_observed_types = len(counts)
    pseudocount = 1.0 / max(total, 1)

    # Compute Shannon entropy with pseudocounts ONLY on observed types
    total_with_pseudo = total + pseudocount * num_observed_types
    entropy = 0.0
    for aa, count in counts.items():
        p = (count + pseudocount) / total_with_pseudo
        if p > 0:
            entropy -= p * math.log2(p)

    # Normalize by log2(20)
    normalized = entropy / LOG2_20
    return round(min(normalized, 1.0), 4)


def classify_conservation(entropy_score: float) -> str:
    """Classify a residue by its entropy score."""
    if entropy_score < 0.2:
        return "highly_conserved"
    elif entropy_score < 0.5:
        return "moderately_conserved"
    else:
        return "variable"


def compute_conservation_entropy(
    query_sequence: str,
    position_alignments: dict[int, list[str]],
) -> dict:
    """
    Compute Shannon entropy conservation scores for each query position.

    Args:
        query_sequence: The query protein sequence.
        position_alignments: Dict mapping 0-based query position → list of
                             amino acids observed at that position across
                             aligned homologs (including the query residue).

    Returns:
        Dictionary with keys:
          - 'entropy_scores': dict {resid_1based: entropy}
          - 'conservation_classes': dict {resid_1based: classification}
          - 'conserved_positions': list of 1-based positions with entropy < 0.2
          - 'summary': human-readable summary
          - 'num_sequences': number of sequences used
    """
    seq_len = len(query_sequence)
    entropy_scores = {}
    conservation_classes = {}
    conserved_positions = []

    # Determine how many sequences were used
    num_seqs = 0
    for pos_aas in position_alignments.values():
        num_seqs = max(num_seqs, len(pos_aas))

    for pos in range(seq_len):
        aas = position_alignments.get(pos, [query_sequence[pos]])
        entropy = compute_entropy(aas)
        resid = pos + 1  # 1-based

        entropy_scores[resid] = entropy
        cls = classify_conservation(entropy)
        conservation_classes[resid] = cls

        if cls == "highly_conserved":
            conserved_positions.append(resid)

    # Count by class
    n_conserved = sum(1 for c in conservation_classes.values() if c == "highly_conserved")
    n_moderate = sum(1 for c in conservation_classes.values() if c == "moderately_conserved")
    n_variable = sum(1 for c in conservation_classes.values() if c == "variable")

    pct_conserved = round(n_conserved / seq_len * 100, 1) if seq_len > 0 else 0

    summary = (
        f"Shannon entropy analysis across {num_seqs} homologs: "
        f"{n_conserved} ({pct_conserved}%) highly conserved, "
        f"{n_moderate} moderately conserved, "
        f"{n_variable} variable positions."
    )

    logger.info("Entropy conservation: %d/%d highly conserved positions", n_conserved, seq_len)

    return {
        "entropy_scores": entropy_scores,
        "conservation_classes": conservation_classes,
        "conserved_positions": conserved_positions,
        "summary": summary,
        "num_sequences": num_seqs,
    }
