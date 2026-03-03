"""
agent.py – ProteusIQ Agent Orchestrator

Central orchestrator class that coordinates all analysis tools and
derived analysis modules. Manages data flow, handles errors gracefully,
and produces the aggregated data dictionary for the UI and report.

Pipeline Steps:
   1. UniProt Metadata
   2. Physicochemical Analysis
   3. Signal Peptide & TM Prediction
   4. BLAST Homology Search
   5. Motif/Domain Detection (PROSITE)
   6. InterPro Domain Search (EBI)
   7. Conservation Analysis (pairwise alignment)
   8. Shannon Entropy Scoring
   9. Structure Search (PDB + AlphaFold)
  10. Ligand Detection
  11. PDB File Download
  12. Ligand Contact Quantification
  13. Spatial Clustering
  14. Secondary Structure
  15. Disorder Prediction
  16. Optional MSA via EBI Clustal Omega
  17. Optional Phylogenetic Tree
  18. Functional Inference
"""

import os
import shutil
import logging

from tools.cache import SequenceCache

from tools import physchem, tm_predict, blast, motif, alignment, structure, ligand
from tools import msa, phylogeny, visualization, uniprot, plots, disorder
from analysis import conservation as entropy_module
from analysis import contacts, clustering, secondary_structure
from reasoning import inference_engine

logger = logging.getLogger(__name__)

# Try importing interpro (optional — may be slow)
try:
    from tools import interpro
    HAS_INTERPRO = True
except ImportError:
    HAS_INTERPRO = False
    logger.info("InterPro module not available")


class ProteinAgent:
    """
    Orchestrates the complete protein analysis pipeline.

    Attributes:
        sequence: The cleaned protein sequence.
        organism: Optional organism name.
        uniprot_id: Optional UniProt accession.
        skip_conservation: If True, skip conservation analysis.
        compute_msa: If True, attempt MSA via EBI.
        build_tree: If True, attempt phylogenetic tree.
        run_interpro: If True, run InterPro domain search.
        warnings: List of warning messages for the UI.
    """

    def __init__(
        self,
        sequence: str,
        organism: str = "",
        uniprot_id: str = "",
        skip_conservation: bool = False,
        compute_msa: bool = False,
        build_tree: bool = False,
        run_interpro: bool = False,
    ):
        self.sequence = sequence
        self.organism = organism
        self.uniprot_id = uniprot_id
        self.skip_conservation = skip_conservation
        self.compute_msa = compute_msa
        self.build_tree = build_tree
        self.run_interpro = run_interpro
        self.warnings = []
        self.data = {}

    def run(self, progress_callback=None) -> dict:
        """
        Execute the full analysis pipeline.

        Args:
            progress_callback: Optional callable(step_name, step_num, total_steps)
                for progress reporting (e.g., Streamlit progress bar).

        Returns:
            Aggregated data dictionary with all analysis results.
        """
        total_steps = 18
        step = 0

        def _progress(step_name: str):
            nonlocal step
            step += 1
            logger.info("Step %d/%d: %s", step, total_steps, step_name)
            if progress_callback:
                progress_callback(step_name, step, total_steps)

        # Initialize results container
        self.data = {
            "sequence": self.sequence,
            "sequence_name": f"Query Protein ({len(self.sequence)} aa)",
            "organism": self.organism,
            "uniprot_id": self.uniprot_id,
        }

        # ─── Setup Content Cache Dir ───
        self.content_dir = os.path.join(os.path.dirname(__file__), ".proteusiq_cache", "contents")
        os.makedirs(self.content_dir, exist_ok=True)

        # ─── Check disk cache ───
        cache = SequenceCache()
        cache_key = cache.make_key(
            self.sequence,
            skip_conservation=self.skip_conservation,
            compute_msa=self.compute_msa,
            build_tree=self.build_tree,
            run_interpro=self.run_interpro,
        )
        cached = cache.get(cache_key)
        if cached:
            logger.info("Cache HIT — returning cached results")
            cached["_from_cache"] = True
            if progress_callback:
                progress_callback("Loaded from cache!", total_steps, total_steps)
            self.data = cached
            self.data["warnings"] = ["Results loaded from cache (analyzed previously)."]
            return self.data

        # ─── Step 1: UniProt Metadata ───
        _progress("UniProt Metadata")
        if self.uniprot_id:
            try:
                self.data["uniprot_meta"] = uniprot.fetch_metadata(self.uniprot_id)
            except Exception as e:
                logger.warning("UniProt fetch failed: %s", e)
                self.data["uniprot_meta"] = {"found": False, "message": str(e)}
        else:
            self.data["uniprot_meta"] = {"found": False, "message": "No UniProt ID provided"}

        # ─── Step 2: Physicochemical Analysis ───
        _progress("Physicochemical Analysis")
        try:
            self.data["physchem"] = physchem.analyze(self.sequence)
        except Exception as e:
            logger.error("Physicochemical analysis failed: %s", e)
            self.warnings.append(f"Physicochemical analysis failed: {e}")
            self.data["physchem"] = {"length": len(self.sequence), "error": str(e)}

        # ─── Step 3: Signal Peptide & TM Prediction ───
        _progress("Signal Peptide & TM Prediction")
        try:
            self.data["tm_prediction"] = tm_predict.predict(self.sequence)
        except Exception as e:
            logger.error("TM prediction failed: %s", e)
            self.warnings.append(f"TM prediction failed: {e}")
            self.data["tm_prediction"] = {
                "signal_peptide": False, "tm_helices": 0,
                "localization": "Unknown", "error": str(e),
            }

        # ─── Step 4: BLAST Homology Search ───
        _progress("BLAST Homology Search")
        if len(self.sequence) > 3000:
            logger.warning("Sequence too long (%d aa), skipping BLAST", len(self.sequence))
            self.warnings.append("BLAST skipped: sequence exceeds 3000 amino acids.")
            self.data["blast"] = {
                "hits": [], "success": False,
                "message": "Skipped (sequence too long)",
            }
        else:
            try:
                def _blast_progress(msg):
                    if progress_callback:
                        progress_callback(msg, step, total_steps)
                self.data["blast"] = blast.search(self.sequence, progress_callback=_blast_progress)
                if not self.data["blast"]["success"]:
                    self.warnings.append(self.data["blast"]["message"])
            except Exception as e:
                logger.error("BLAST search failed: %s", e)
                self.warnings.append(f"BLAST search failed: {e}")
                self.data["blast"] = {
                    "hits": [], "success": False,
                    "message": f"Error: {e}",
                }

        # ─── Step 5: Motif/Domain Detection (PROSITE) ───
        _progress("Motif & Domain Detection")
        try:
            self.data["domains"] = motif.scan(self.sequence)
        except Exception as e:
            logger.error("Motif scanning failed: %s", e)
            self.warnings.append(f"Motif scanning failed: {e}")
            self.data["domains"] = []

        # ─── Step 6: InterPro Domain Search ───
        _progress("InterPro Domain Search")
        if self.run_interpro and HAS_INTERPRO:
            try:
                self.data["interpro"] = interpro.search_domains(self.sequence)
                if not self.data["interpro"]["success"]:
                    self.warnings.append(
                        f"InterPro: {self.data['interpro'].get('message', 'failed')}"
                    )
            except Exception as e:
                logger.error("InterPro search failed: %s", e)
                self.data["interpro"] = {
                    "domains": [], "families": [], "go_terms": [],
                    "success": False, "message": str(e),
                }
        else:
            self.data["interpro"] = {
                "domains": [], "families": [], "go_terms": [],
                "success": False,
                "message": "InterPro search not requested." if not self.run_interpro else "InterPro module unavailable.",
            }

        # ─── Step 7: Conservation Analysis (pairwise alignment) ───
        _progress("Conservation Analysis")
        blast_hits = self.data.get("blast", {}).get("hits", [])

        if self.skip_conservation or len(self.sequence) > 3000:
            logger.info("Conservation analysis skipped")
            self.data["conservation"] = {
                "conserved_positions": [],
                "conservation_scores": [],
                "position_alignments": {},
                "hit_sequences": {},
                "summary": "Conservation analysis was skipped.",
                "num_sequences": 0,
                "skipped": True,
                "message": "Skipped",
            }
        else:
            try:
                self.data["conservation"] = alignment.compute_conservation(
                    self.sequence, blast_hits
                )
            except Exception as e:
                logger.error("Conservation analysis failed: %s", e)
                self.warnings.append(f"Conservation analysis failed: {e}")
                self.data["conservation"] = {
                    "conserved_positions": [],
                    "conservation_scores": [],
                    "position_alignments": {},
                    "hit_sequences": {},
                    "summary": f"Conservation analysis failed: {e}",
                    "num_sequences": 0,
                    "skipped": True,
                    "message": str(e),
                }

        # ─── Step 8: Shannon Entropy Scoring ───
        _progress("Shannon Entropy Conservation")
        pos_alignments = self.data.get("conservation", {}).get("position_alignments", {})

        if pos_alignments and not self.data["conservation"].get("skipped"):
            try:
                self.data["entropy"] = entropy_module.compute_conservation_entropy(
                    self.sequence, pos_alignments
                )
            except Exception as e:
                logger.error("Entropy scoring failed: %s", e)
                self.warnings.append(f"Entropy scoring failed: {e}")
                self.data["entropy"] = {
                    "entropy_scores": {}, "conservation_classes": {},
                    "conserved_positions": [], "summary": str(e),
                }
        else:
            self.data["entropy"] = {
                "entropy_scores": {}, "conservation_classes": {},
                "conserved_positions": [],
                "summary": "Entropy scoring skipped (conservation data not available).",
            }

        # ─── Step 9: Structure Search ───
        _progress("Structure Identification")
        try:
            self.data["structure"] = structure.search(
                self.sequence,
                uniprot_id=self.uniprot_id,
                blast_hits=blast_hits,
            )
        except Exception as e:
            logger.error("Structure search failed: %s", e)
            self.warnings.append(f"Structure search failed: {e}")
            self.data["structure"] = {"found": False, "message": str(e)}

        # ─── Step 10: Ligand Detection ───
        _progress("Ligand Detection")
        struct_data = self.data.get("structure", {})
        pdb_id = struct_data.get("pdb_id", "")

        if pdb_id and struct_data.get("source") == "PDB":
            try:
                self.data["ligands"] = ligand.detect_ligands(pdb_id)
            except Exception as e:
                logger.error("Ligand detection failed: %s", e)
                self.warnings.append(f"Ligand detection failed: {e}")
                self.data["ligands"] = {
                    "ligands": [], "total_unique": 0, "message": str(e),
                }
        else:
            self.data["ligands"] = {
                "ligands": [], "total_unique": 0, "filtered": False,
                "message": "No PDB structure available for ligand detection.",
            }

        # ─── Step 11: Download PDB File ───
        _progress("Downloading Structure Data")
        pdb_path = ""

        if struct_data.get("found"):
            try:
                if struct_data.get("source") == "PDB" and pdb_id:
                    pdb_path = structure.download_pdb_file(pdb_id=pdb_id)
                elif struct_data.get("source") == "AlphaFold" and struct_data.get("pdb_url"):
                    pdb_path = structure.download_pdb_file(pdb_url=struct_data["pdb_url"])
            except Exception as e:
                logger.warning("PDB file download failed: %s", e)

        self.data["pdb_path"] = pdb_path

        # Read PDB content for 3D viewer, save to disk instead of session_state
        pdb_content_file = ""
        if pdb_path and os.path.exists(pdb_path):
            try:
                seq_hash = hashlib.sha256(self.sequence.encode()).hexdigest()[:16]
                ext = ".cif" if str(pdb_path).strip().lower().endswith(".cif") else ".pdb"
                pdb_content_file = os.path.join(self.content_dir, f"pdb_{seq_hash}{ext}")
                
                # Copy file to persistent content cache
                shutil.copy2(pdb_path, pdb_content_file)
            except Exception as e:
                logger.warning("Failed to save PDB content to disk: %s", e)
                
        self.data["pdb_content_file"] = pdb_content_file

        # ─── Step 12: Ligand Contact Quantification ───
        _progress("Ligand Contact Analysis")
        entropy_scores = self.data.get("entropy", {}).get("entropy_scores", {})
        cons_classes = self.data.get("entropy", {}).get("conservation_classes", {})

        if pdb_path and self.data["ligands"].get("ligands"):
            try:
                self.data["ligand_contacts"] = contacts.quantify_contacts(
                    pdb_path,
                    entropy_scores=entropy_scores,
                    conservation_classes=cons_classes,
                )
            except Exception as e:
                logger.error("Ligand contact analysis failed: %s", e)
                self.data["ligand_contacts"] = {
                    "ligand_contacts": [], "total_ligands": 0, "message": str(e),
                }
        else:
            self.data["ligand_contacts"] = {
                "ligand_contacts": [], "total_ligands": 0,
                "message": "Ligand contact analysis skipped (no structure or ligands).",
            }

        # ─── Step 13: Spatial Clustering ───
        _progress("Spatial Clustering Analysis")
        if pdb_path and entropy_scores:
            try:
                self.data["clustering"] = clustering.cluster_conserved(
                    pdb_path, entropy_scores
                )
            except Exception as e:
                logger.error("Clustering analysis failed: %s", e)
                self.data["clustering"] = {
                    "residues": [], "num_residues": 0,
                    "mean_distance": 0, "p_value": 1.0,
                    "significant": False, "message": str(e),
                }
        else:
            self.data["clustering"] = {
                "residues": [], "num_residues": 0,
                "mean_distance": 0, "p_value": 1.0,
                "significant": False,
                "message": "Clustering skipped (requires structure + conservation).",
            }

        # ─── Step 14: Secondary Structure ───
        _progress("Secondary Structure")
        if pdb_path:
            try:
                self.data["secondary_structure"] = secondary_structure.parse_secondary_structure(
                    pdb_path
                )
            except Exception as e:
                logger.error("Secondary structure parsing failed: %s", e)
                self.data["secondary_structure"] = {
                    "assignments": {}, "summary": str(e),
                }
        else:
            self.data["secondary_structure"] = {
                "assignments": {}, "helix_ranges": [], "sheet_ranges": [],
                "summary": "Secondary structure not available (requires PDB structure).",
            }

        # ─── Cleanup PDB temp file (after all PDB-dependent steps) ───
        if pdb_path and os.path.exists(pdb_path):
            try:
                pdb_dir = os.path.dirname(pdb_path)
                if "proteusiq_" in pdb_dir:
                    shutil.rmtree(pdb_dir, ignore_errors=True)
                    logger.info("Cleaned up temp PDB directory: %s", pdb_dir)
                    self.data["pdb_path"] = ""  # clear stale ref
            except Exception as e:
                logger.debug("PDB cleanup failed: %s", e)

        # ─── Step 15: Disorder Prediction ───
        _progress("Disorder Prediction")
        try:
            self.data["disorder"] = disorder.predict(self.sequence)
        except Exception as e:
            logger.error("Disorder prediction failed: %s", e)
            self.data["disorder"] = {
                "scores": {}, "disordered_regions": [],
                "disorder_content": 0.0, "num_disordered": 0,
                "summary": f"Disorder prediction failed: {e}",
            }

        # ─── Step 16: Optional MSA ───
        _progress("Multiple Sequence Alignment")
        hit_seqs = self.data.get("conservation", {}).get("hit_sequences", {})

        if self.compute_msa and len(hit_seqs) >= 2:  # query + 2 hits = 3 total (Clustal Omega min)
            try:
                self.data["msa"] = msa.run_clustalo(self.sequence, hit_seqs)
            except Exception as e:
                logger.error("MSA failed: %s", e)
                self.data["msa"] = {
                    "alignment": "", "success": False, "message": str(e),
                }
        else:
            msg = "MSA not requested." if not self.compute_msa else f"Fewer than 3 sequences available (have {1 + len(hit_seqs)})."
            self.data["msa"] = {
                "alignment": "", "success": False, "message": msg,
            }

        # ─── Step 17: Optional Phylogeny ───
        _progress("Phylogenetic Tree")
        msa_data = self.data.get("msa", {})

        if (self.build_tree and msa_data.get("success")
                and msa_data.get("num_sequences", 0) >= 4):
            try:
                self.data["phylogeny"] = phylogeny.build_tree(msa_data["alignment"])
            except Exception as e:
                logger.error("Tree construction failed: %s", e)
                self.data["phylogeny"] = {
                    "newick": "", "ascii_tree": "", "success": False, "message": str(e),
                }
        else:
            msg = "Phylogeny not requested." if not self.build_tree else "MSA not available or <4 sequences."
            self.data["phylogeny"] = {
                "newick": "", "ascii_tree": "", "success": False, "message": msg,
            }

        # ─── Step 18: Functional Inference ───
        _progress("Functional Inference")
        try:
            self.data["inference"] = inference_engine.infer_function(self.data)
        except Exception as e:
            logger.error("Functional inference failed: %s", e)
            self.warnings.append(f"Functional inference failed: {e}")
            self.data["inference"] = {
                "function": "Unable to infer function",
                "confidence": "Low",
                "rules_triggered": [],
                "evidence_count": 0,
            }

        # Store warnings in data for UI
        self.data["warnings"] = self.warnings

        logger.info("Analysis pipeline complete. %d warnings.", len(self.warnings))

        # ─── Write to disk cache ───
        try:
            # Strip pdb_content_file from disk cache (we only recreate it on live runs)
            # Actually, we can keep the file path in cache, since the file is persistent in .proteusiq_cache/contents
            cache.set(cache_key, self.data)
        except Exception as e:
            logger.debug("Cache write failed: %s", e)

        return self.data
