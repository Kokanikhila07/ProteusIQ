"""
app.py – ProteusIQ Streamlit Frontend

Protein Sequence & Structural Analysis platform.
Tabbed interface with session state persistence, premium design,
and comprehensive protein analysis capabilities.

Tabs:
  1. Overview     – UniProt metadata, physicochemical, composition chart
  2. Structure    – 3D viewer, secondary structure, ligand contacts
  3. Evolution    – Conservation, clustering, BLAST, MSA, phylogeny
  4. Advanced     – Sequence viewer, hydropathy plot, inference, export
"""

import os
import json
import base64
import logging
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from agent import ProteinAgent
from report.generate_report import generate_html
from tools.visualization import render_advanced_viewer
from tools.plots import (
    compute_hydropathy,
    generate_sequence_annotation_html,
    generate_conservation_bar_html,
)
from tools import structure

# ─── Logging setup ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Valid amino acids ───
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# ─── Example sequences ───
EXAMPLES = {
    "— Select an example —": ("", "", ""),
    "EGFR (Human, 1210 aa)": (
        """>sp|P00533|EGFR_HUMAN
MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVV
LGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVL
SNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVS SDFLSNMSMDFQN
HLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGP
RESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDH
GSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSIS
GDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIR
GRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKT
KIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREF
VENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKY
ADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPSIATGMVGALLLLLVVALGIGLFM""",
        "Homo sapiens", "P00533",
    ),
    "Human Serum Albumin (609 aa)": (
        """>sp|P02768|ALBU_HUMAN
MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFE
DHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERN
ECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAK
RYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQ
RFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPL
LEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVL
LLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNAL
LVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTP
VSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALV
ELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL""",
        "Homo sapiens", "P02768",
    ),
    "LacY Lactose Permease (E. coli, 417 aa)": (
        """>sp|P02920|LACY_ECOLI
MYYLKNTNFWMFGLFFFFYFFIMGAYFPFFPIWLHDINHISKSDTGIIFAAIS LFSLLFQL
FRNMYGLTAGQLLSIFKAANILHTHVWAHFYAWMSSIKDAMDLGQEIDDNILAQFGNIAF
AHFHNLAWGVAGLSGAHAFIFMRVPFQTFSALVRFLTAASLLTLNLVTAENINAILLFYNH
QLHKSPAAIALTIAHLCCNASALTGVIWLGTNYGWTFHKNPFSFIAGIAVLASSMLFIPIK
KVVFMPQTIRFAVKFAQSKMIYAGLVGMCFASALWLLFKKSPIHFVDRPFSEYDSRADTFD
PEFIQLEHNPSASPRWFLLRGKPAISAGLLYANYGYFLAHSQFHASQAHTNLSQGFRSHMA
FDFPERQPIQVTLLDIAHKLEGMVAGVTVIISLLMNTLPFMTLKMVYQLAGG""",
        "Escherichia coli", "P02920",
    ),
    "Lysozyme (Human, 148 aa)": (
        """>sp|P61626|LYSC_HUMAN
MKALIVLGLVLLSVTVQGKVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRTN
YNAGDRSTDYGIFQINSRYWCNDGKTPGAVNACHLSCSALLQDNIADAVACAKRVVRDPQG
IRAWVAWRNRCQNRDVRQYVQGCGV""",
        "Homo sapiens", "P61626",
    ),
}


def _parse_fasta_and_metadata(raw_input: str) -> tuple[str, str, str]:
    """Parse FASTA input: extract sequence, UniProt ID, and organism if present."""
    import re
    lines = raw_input.strip().split("\n")
    seq_lines = []
    extracted_uniprot = ""
    extracted_organism = ""
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(">"):
            # Try to extract UniProt ID: >sp|P00533|... or >tr|Q9H2C0|...
            uniprot_match = re.search(r'>(?:sp|tr)\|([A-Z0-9]+)\|', stripped)
            if uniprot_match:
                extracted_uniprot = uniprot_match.group(1)
            
            # Try to extract Organism: OS=Homo sapiens OV=...
            org_match = re.search(r'OS=([a-zA-Z0-9 ]+?)(?:\s+[A-Z]{2}=|$)', stripped)
            if org_match:
                extracted_organism = org_match.group(1).strip()
            continue
        seq_lines.append(stripped)
        
    sequence = "".join(seq_lines).upper().replace(" ", "")
    return sequence, extracted_uniprot, extracted_organism


def _validate_sequence(sequence: str) -> tuple:
    """Validate the protein sequence. Returns (is_valid, cleaned, error)."""
    if not sequence:
        return False, "", "Please enter a protein sequence."

    if len(sequence) < 10:
        return False, sequence, "Sequence too short (minimum 10 amino acids)."

    invalid_chars = set(sequence) - VALID_AA
    if invalid_chars:
        return (
            False, sequence,
            f"Invalid character(s): **{', '.join(sorted(invalid_chars))}**. "
            f"Only the 20 standard amino acids are accepted."
        )

    return True, sequence, ""


def main():
    """Main Streamlit application."""

    # ─── Page configuration ───
    st.set_page_config(
        page_title="ProteusIQ – Protein Sequence & Structural Analysis",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ─── Premium CSS ───
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        .stApp { font-family: 'Inter', sans-serif; }

        /* ── Header ── */
        .main-header {
            text-align: center;
            padding: 1rem 0 0.3rem 0;
        }
        .main-header h1 {
            background: linear-gradient(135deg, #7c9dff, #c084fc, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0;
        }
        .main-header p {
            color: #9ca3af;
            font-size: 1rem;
            margin-top: 0;
        }

        /* ── Glass cards ── */
        .glass-card {
            background: linear-gradient(135deg,
                rgba(124,157,255,0.06), rgba(192,132,252,0.06));
            backdrop-filter: blur(12px);
            border: 1px solid rgba(124,157,255,0.12);
            border-radius: 14px;
            padding: 1.2rem;
            margin-bottom: 0.8rem;
            transition: border-color 0.2s ease;
        }
        .glass-card:hover {
            border-color: rgba(124,157,255,0.25);
        }
        .glass-card h4 {
            margin: 0 0 0.6rem 0;
            color: #c8d6ff;
            font-size: 0.95rem;
            font-weight: 600;
        }

        /* ── UniProt card ── */
        .uniprot-card {
            background: linear-gradient(135deg,
                rgba(0,200,150,0.08), rgba(0,150,255,0.06));
            border: 1px solid rgba(0,200,150,0.18);
            border-radius: 14px;
            padding: 1.2rem;
            margin-bottom: 0.8rem;
        }

        /* ── Stat pills ── */
        .stat-pill {
            display: inline-block;
            background: rgba(124,157,255,0.1);
            border: 1px solid rgba(124,157,255,0.15);
            border-radius: 8px;
            padding: 6px 14px;
            margin: 3px;
            font-size: 0.85rem;
            color: #d0d8ff;
        }
        .stat-pill strong { color: #a5b4fc; }

        /* ── Cluster badges ── */
        .badge-yes {
            background: rgba(0,255,136,0.12);
            border: 1px solid rgba(0,255,136,0.3);
            border-radius: 8px;
            padding: 0.6rem 1rem;
            color: #00ff88;
        }
        .badge-no {
            background: rgba(255,165,0,0.1);
            border: 1px solid rgba(255,165,0,0.25);
            border-radius: 8px;
            padding: 0.6rem 1rem;
            color: #ffa500;
        }

        /* ── Tabs styling ── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 8px 20px;
        }

        /* ── Progress area ── */
        .analysis-running {
            background: linear-gradient(135deg,
                rgba(124,157,255,0.05), rgba(192,132,252,0.05));
            border: 1px solid rgba(124,157,255,0.1);
            border-radius: 12px;
            padding: 1.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # ─── Header ───
    st.markdown("""
    <div class="main-header">
        <h1>🧬 ProteusIQ</h1>
        <p>Protein Sequence & Structural Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # ─── Sidebar ───
    with st.sidebar:
        st.markdown("### 🔬 Input")

        # Example selector
        example_choice = st.selectbox(
            "Load example:", list(EXAMPLES.keys()),
            help="Pre-loaded protein sequences for quick testing"
        )
        example_seq, example_org, example_uid = EXAMPLES.get(example_choice, ("", "", ""))

        sequence_input = st.text_area(
            "Protein Sequence (FASTA or raw)",
            value=example_seq,
            height=160,
            placeholder="Paste your protein sequence here...",
        )

        col1, col2 = st.columns(2)
        with col1:
            organism = st.text_input("Organism", value=example_org, placeholder="e.g., Homo sapiens")
        with col2:
            uniprot_id = st.text_input("UniProt ID", value=example_uid, placeholder="e.g., P00533")

        st.markdown("---")
        st.markdown("##### ⚙️ Options")
        filter_ions = st.checkbox("Filter crystallization ions", value=False)

        st.markdown("##### 🧪 Optional Advanced")
        compute_msa = st.checkbox("Compute MSA (EBI Clustal Omega)", value=False,
                                   help="Requires ≥3 sequences. May take 1-3 minutes.")
        build_tree = st.checkbox("Build phylogenetic tree", value=False,
                                  help="Requires MSA + ≥4 sequences. Illustrative only.")
        run_interpro = st.checkbox("Run InterPro domain search", value=False,
                                    help="Rich domain/family annotations via EBI. May take 5-10 minutes.")

        st.markdown("---")
        analyze_button = st.button("🚀 Analyze Protein", type="primary", use_container_width=True)

    # ─── Session state for results persistence ───
    if "analysis_data" not in st.session_state:
        st.session_state.analysis_data = None

    # ─── Run analysis ───
    if analyze_button and sequence_input:
        sequence, extracted_uid, extracted_org = _parse_fasta_and_metadata(sequence_input)
        
        # Override empty sidebar fields with extracted data if available
        if not uniprot_id and extracted_uid:
            uniprot_id = extracted_uid
        if not organism and extracted_org:
            organism = extracted_org

        is_valid, sequence, error_msg = _validate_sequence(sequence)

        if not is_valid:
            st.error(error_msg)
            return

        # Determine if conservation should be skipped
        skip_conservation = len(sequence) > 2500

        if skip_conservation:
            st.warning(
                f"⚠️ Long sequence ({len(sequence)} aa) — conservation analysis "
                "will be skipped for performance."
            )

        st.info(f"🔬 Analyzing **{len(sequence)} amino acids**...")

        # Progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(step_name, step_num, total):
            progress_bar.progress(step_num / total)
            status_text.text(f"Step {step_num}/{total}: {step_name}")

        agent = ProteinAgent(
            sequence=sequence,
            organism=organism,
            uniprot_id=uniprot_id,
            skip_conservation=skip_conservation,
            compute_msa=compute_msa,
            build_tree=build_tree,
            run_interpro=run_interpro,
        )

        try:
            data = agent.run(progress_callback=progress_callback)
            st.session_state.analysis_data = data
        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")
            logger.exception("Pipeline failure")
            return

        progress_bar.progress(1.0)
        status_text.text("✅ Analysis complete!")

        # Show cache indicator
        if data.get("_from_cache"):
            st.success("⚡ Results loaded from cache — no API calls needed!")

    # ─── Display results from session state ───
    data = st.session_state.analysis_data

    if data is None:
        _show_landing_page()
        return

    # Show warnings
    for w in data.get("warnings", []):
        st.warning(f"⚠️ {w}")

    # ═══════════════════════════════════════════════════════
    # TABBED INTERFACE
    # ═══════════════════════════════════════════════════════
    tab_overview, tab_structure, tab_evolution, tab_advanced = st.tabs([
        "📊 Overview", "🏗️ Structure", "📈 Evolution", "🔬 Advanced"
    ])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 1: OVERVIEW
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_overview:
        # UniProt metadata card
        uni = data.get("uniprot_meta", {})
        if uni.get("found"):
            st.markdown(f"""
            <div class="uniprot-card">
                <h4>🟢 UniProt: {data.get('uniprot_id', '')}</h4>
                <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:8px;">
                    <span class="stat-pill"><strong>Gene:</strong> {uni.get('gene_name', 'N/A')}</span>
                    <span class="stat-pill"><strong>Protein:</strong> {uni.get('protein_name', 'N/A')}</span>
                    <span class="stat-pill"><strong>Organism:</strong> {uni.get('organism', 'N/A')}</span>
                </div>
                {"<p style='color:#b0bfcf;font-size:0.9rem;margin:0;'><b>Function:</b> " + uni.get('function', '') + "</p>" if uni.get('function') else ""}
            </div>
            """, unsafe_allow_html=True)

            # Subcellular location from UniProt
            if uni.get("subcellular_location"):
                locs = ", ".join(uni["subcellular_location"])
                st.caption(f"📍 UniProt subcellular location: {locs}")

            # Keywords
            if uni.get("keywords"):
                kw_str = " · ".join(uni["keywords"][:10])
                st.caption(f"🏷️ Keywords: {kw_str}")

        # ─── Physicochemical Properties ───
        st.markdown("### 📊 Physicochemical Properties")
        pc = data.get("physchem", {})
        if "error" not in pc:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Length", f"{pc['length']} aa")
            c2.metric("Molecular Weight", f"{pc['molecular_weight']/1000:.1f} kDa")
            c3.metric("Theoretical pI", pc["theoretical_pI"])
            stability = "Stable" if pc["instability_index"] <= 40 else "Unstable"
            c4.metric("Stability", f"{stability} ({pc['instability_index']:.1f})")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("GRAVY", pc["gravy"])
            c6.metric("Aromaticity", pc["aromaticity"])

            # Localization
            tm = data.get("tm_prediction", {})
            c7.metric("Localization", tm.get("localization", "?"))
            c8.metric("TM Helices", tm.get("tm_helices", 0))

            # Amino acid composition chart
            if pc.get("amino_acid_percent"):
                with st.expander("📊 Amino Acid Composition"):
                    aa_data = pc["amino_acid_percent"]
                    aa_df = pd.DataFrame({
                        "Amino Acid": list(aa_data.keys()),
                        "Percentage": [round(v * 100, 2) for v in aa_data.values()],
                    }).set_index("Amino Acid")
                    st.bar_chart(aa_df, height=250)
        else:
            st.error(pc["error"])

        # ─── Functional Inference ───
        st.markdown("### 🧠 Functional Inference")
        inf = data.get("inference", {})
        conf = inf.get("confidence", "Low")
        conf_icon = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(conf, "⚪")

        st.markdown(f"""
        <div class="glass-card">
            <h4>{conf_icon} {inf.get('function', 'Unable to infer function')}</h4>
            <span class="stat-pill"><strong>Confidence:</strong> {conf}</span>
            <span class="stat-pill"><strong>Evidence:</strong> {inf.get('evidence_count', 0)} rule(s)</span>
        </div>
        """, unsafe_allow_html=True)

        if inf.get("rules_triggered"):
            with st.expander("View evidence rules"):
                for rule in inf["rules_triggered"]:
                    st.markdown(f"- {rule}")

        # ─── BLAST Hits ───
        st.markdown("### 🔍 Homology Search")
        blast_data = data.get("blast", {})
        if blast_data.get("success") and blast_data.get("hits"):
            st.success(blast_data["message"])
            hits_df = pd.DataFrame(blast_data["hits"])
            display_cols = ["accession", "definition", "organism",
                           "identity_pct", "coverage_pct", "evalue"]
            available = [c for c in display_cols if c in hits_df.columns]
            st.dataframe(
                hits_df[available].rename(columns={
                    "accession": "Accession", "definition": "Description",
                    "organism": "Organism", "identity_pct": "Identity %",
                    "coverage_pct": "Coverage %", "evalue": "E-value",
                }),
                use_container_width=True, hide_index=True,
            )
        elif blast_data.get("success"):
            st.info("No significant homologs found (≥30% identity).")
        else:
            st.warning(blast_data.get("message", "BLAST did not complete."))

        # Domains (PROSITE)
        st.markdown("### 🧩 Motifs & Domains")
        domains = data.get("domains", [])
        if domains:
            st.markdown("**PROSITE Pattern Matches:**")
            dom_df = pd.DataFrame(domains)
            st.dataframe(
                dom_df.rename(columns={
                    "id": "PROSITE ID", "name": "Name",
                    "start": "Start", "end": "End",
                }),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No known PROSITE motifs detected.")

        # InterPro domains
        interpro_data = data.get("interpro", {})
        if interpro_data.get("success"):
            ipr_domains = interpro_data.get("domains", [])
            ipr_families = interpro_data.get("families", [])

            if ipr_domains:
                st.markdown("**InterPro Domain Hits:**")
                ipr_df = pd.DataFrame(ipr_domains)
                display_cols = ["db", "signature_id", "signature_name",
                               "start", "end", "ipr_id", "ipr_name"]
                available = [c for c in display_cols if c in ipr_df.columns]
                st.dataframe(
                    ipr_df[available].rename(columns={
                        "db": "Database", "signature_id": "Signature",
                        "signature_name": "Name", "start": "Start",
                        "end": "End", "ipr_id": "InterPro",
                        "ipr_name": "InterPro Name",
                    }),
                    use_container_width=True, hide_index=True,
                )

            if ipr_families:
                st.markdown("**InterPro Family/Superfamily:**")
                fam_df = pd.DataFrame(ipr_families)
                display_cols = ["db", "signature_id", "signature_name",
                               "ipr_id", "ipr_name"]
                available = [c for c in display_cols if c in fam_df.columns]
                st.dataframe(
                    fam_df[available].rename(columns={
                        "db": "Database", "signature_id": "Signature",
                        "signature_name": "Name",
                        "ipr_id": "InterPro", "ipr_name": "Family",
                    }),
                    use_container_width=True, hide_index=True,
                )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 2: STRUCTURE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_structure:
        struct = data.get("structure", {})
        pdb_content = data.get("pdb_content", "")

        # Structure info
        if struct.get("found"):
            if struct.get("source") == "PDB":
                c1, c2, c3 = st.columns(3)
                c1.metric("PDB ID", struct["pdb_id"])
                c2.metric("Method", struct.get("method", "?"))
                res = struct.get("resolution")
                c3.metric("Resolution", f"{res} Å" if res else "N/A")
                if struct.get("title"):
                    st.caption(struct["title"])
            elif struct.get("source") == "AlphaFold":
                c1, c2, c3 = st.columns(3)
                c1.metric("Source", "AlphaFold")
                c2.metric("UniProt", struct.get("uniprot_id", ""))
                plddt = struct.get("avg_plddt")
                c3.metric("Avg. pLDDT",
                           f"{plddt}" if plddt else "N/A")

            # ─── 3D Viewer ───
            pdb_content = data.get("pdb_content", "")
            pdb_content_file = ""
            if not pdb_content:
                pdb_content_file = data.get("pdb_content_file", "")
                
                # If cached file is missing or path is empty, re-download on-the-fly
                if not pdb_content_file or not os.path.exists(pdb_content_file):
                    if struct.get("source") == "PDB" and struct.get("pdb_id"):
                        pdb_content_file = structure.download_pdb_file(pdb_id=struct["pdb_id"])
                    elif struct.get("source") == "AlphaFold" and struct.get("pdb_url"):
                        pdb_content_file = structure.download_pdb_file(pdb_url=struct["pdb_url"])

                # Load from disk on-demand to save session_state RAM
                if pdb_content_file and os.path.exists(pdb_content_file):
                    try:
                        with open(pdb_content_file, "r") as f:
                            pdb_content = f.read()
                    except Exception as e:
                        st.error(f"Failed to load structure content: {e}")

            if pdb_content:
                st.markdown("### 🏗️ Interactive 3D Viewer")
                entropy_scores = data.get("entropy", {}).get("entropy_scores", {})
                ss_assign = data.get("secondary_structure", {}).get("assignments", {})
                tm_positions = data.get("tm_prediction", {}).get("tm_helix_positions", [])
                cluster_res = data.get("clustering", {}).get("residues", [])

                all_contacts = []
                for lig in data.get("ligand_contacts", {}).get("ligand_contacts", []):
                    all_contacts.extend(lig.get("contacts", []))

                # Auto-detect format from file extension, default to pdb if unknown
                fmt = "cif" if pdb_content_file.lower().endswith(".cif") else "pdb"

                viewer_html = render_advanced_viewer(
                    pdb_data=pdb_content,
                    pdb_format=fmt,
                    conservation_dict=entropy_scores or None,
                    ligand_contacts=all_contacts or None,
                    tm_helices=tm_positions or None,
                    secondary_structure=ss_assign or None,
                    cluster_residues=cluster_res or None,
                )
                components.html(viewer_html, height=620, scrolling=False)

                st.caption(
                    "**Controls:** Left-click drag = rotate · Right-click drag = translate · "
                    "Scroll = zoom · Hover = residue info"
                )
            else:
                if struct.get("source") == "PDB":
                    st.markdown(
                        f"[View on RCSB PDB →](https://www.rcsb.org/structure/{struct['pdb_id']})"
                    )
        else:
            st.info("No experimental or predicted structure available for this sequence.")

        # ─── Secondary Structure ───
        st.markdown("### 🔬 Secondary Structure")
        ss = data.get("secondary_structure", {})
        if ss.get("assignments"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Helix", f"{ss.get('helix_pct', 0)}%")
            c2.metric("Sheet", f"{ss.get('sheet_pct', 0)}%")
            c3.metric("Coil", f"{ss.get('coil_pct', 0)}%")

            if ss.get("ss_string"):
                with st.expander("Per-residue assignment"):
                    ss_str = ss["ss_string"]
                    for i in range(0, len(ss_str), 80):
                        st.code(ss_str[i:i+80], language=None)
        else:
            st.info(ss.get("summary", "Secondary structure not available (requires PDB)."))

        # ─── Ligands & Contacts ───
        st.markdown("### 💊 Ligands & Contacts")
        lig_data = data.get("ligands", {})
        contact_data = data.get("ligand_contacts", {})

        if lig_data.get("ligands"):
            st.success(lig_data["message"])
            lig_df = pd.DataFrame(lig_data["ligands"])
            st.dataframe(
                lig_df.rename(columns={"name": "Ligand", "chain": "Chain", "count": "Count"}),
                use_container_width=True, hide_index=True,
            )

            if contact_data.get("ligand_contacts"):
                for lc in contact_data["ligand_contacts"]:
                    st.markdown(f"**{lc['summary']}**")
                    if lc.get("contacts"):
                        cdf = pd.DataFrame(lc["contacts"])
                        cols = ["resid", "aa", "chain", "distance",
                               "conservation_class", "is_catalytic"]
                        avail = [c for c in cols if c in cdf.columns]
                        st.dataframe(
                            cdf[avail].rename(columns={
                                "resid": "Res#", "aa": "AA", "chain": "Chain",
                                "distance": "Dist (Å)", "conservation_class": "Conservation",
                                "is_catalytic": "Catalytic?",
                            }),
                            use_container_width=True, hide_index=True,
                        )
        else:
            st.info(lig_data.get("message", "No ligands detected."))

        # ─── Structural Confidence ───
        with st.expander("🎯 Structural Confidence Assessment"):
            if struct.get("found") and struct.get("source") == "PDB":
                st.success(
                    f"Experimental: **{struct['pdb_id']}** ({struct.get('method', '')})"
                )
            elif struct.get("found") and struct.get("source") == "AlphaFold":
                plddt = struct.get("avg_plddt")
                st.info(f"AlphaFold model (pLDDT: {plddt}) — {struct.get('interpretation', '')}")
                st.markdown("""
                | pLDDT | Interpretation |
                |---|---|
                | ≥ 90 | Very high confidence |
                | 70–90 | Confident |
                | 50–70 | Low confidence |
                | < 50 | Likely disordered |
                """)
            else:
                st.warning("No structure available.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 3: EVOLUTION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_evolution:
        # ─── Shannon Entropy Conservation ───
        st.markdown("### 📈 Conservation (Shannon Entropy)")
        entropy_data = data.get("entropy", {})

        if entropy_data.get("entropy_scores"):
            st.markdown(f"**{entropy_data.get('summary', '')}**")

            classes = entropy_data.get("conservation_classes", {})
            n_high = sum(1 for v in classes.values() if v == "highly_conserved")
            n_mod = sum(1 for v in classes.values() if v == "moderately_conserved")
            n_var = sum(1 for v in classes.values() if v == "variable")

            c1, c2, c3 = st.columns(3)
            c1.metric("🔴 Highly Conserved", f"{n_high}", help="Entropy < 0.2")
            c2.metric("🟡 Moderate", f"{n_mod}", help="0.2 ≤ Entropy < 0.5")
            c3.metric("🔵 Variable", f"{n_var}", help="Entropy ≥ 0.5")

            # Conservation heatmap bar
            bar_html = generate_conservation_bar_html(
                entropy_data["entropy_scores"],
                len(data.get("sequence", "")),
            )
            if bar_html:
                st.markdown(bar_html, unsafe_allow_html=True)
        else:
            cons_data = data.get("conservation", {})
            st.info(cons_data.get("summary", "Conservation analysis not performed."))

        # ─── Spatial Clustering ───
        st.markdown("### 🎯 Spatial Clustering")
        cluster = data.get("clustering", {})
        if cluster.get("residues"):
            if cluster.get("significant"):
                st.markdown(
                    f'<div class="badge-yes">✅ <b>Significant spatial cluster!</b> '
                    f'{cluster["message"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="badge-no">⚠️ {cluster["message"]}</div>',
                    unsafe_allow_html=True
                )

            c1, c2, c3 = st.columns(3)
            c1.metric("Cluster Size", cluster["num_residues"])
            c2.metric("Mean Distance", f"{cluster['mean_distance']} Å")
            c3.metric("p-value", f"{cluster['p_value']:.4f}")

            with st.expander("Cluster residues"):
                clust_df = pd.DataFrame(cluster["residues"])
                st.dataframe(
                    clust_df.rename(columns={
                        "resid": "Res#", "aa": "AA", "entropy": "Entropy"
                    }),
                    use_container_width=True, hide_index=True,
                )
        else:
            st.info(cluster.get("message", "Clustering not performed."))

        # ─── MSA ───
        st.markdown("### 📐 Multiple Sequence Alignment")
        msa_data = data.get("msa", {})
        if msa_data.get("success") and msa_data.get("alignment"):
            st.success(msa_data["message"])
            if msa_data.get("disclaimer"):
                st.caption(f"ℹ️ {msa_data['disclaimer']}")
            with st.expander("View alignment", expanded=True):
                st.code(msa_data["alignment"], language=None)
        else:
            st.info(msa_data.get("message", "MSA not performed."))

        # ─── Phylogeny ───
        st.markdown("### 🌳 Phylogenetic Tree")
        phylo_data = data.get("phylogeny", {})
        if phylo_data.get("success"):
            st.success(phylo_data["message"])
            st.info(f"ℹ️ {phylo_data.get('disclaimer', '')}")

            # Display Matplotlib-rendered tree image (preferred)
            tree_image = phylo_data.get("tree_image", "")
            if tree_image:
                import base64 as b64
                image_bytes = b64.b64decode(tree_image)
                st.image(image_bytes, caption="Phylogenetic tree with bootstrap support values",
                         use_container_width=True)
            elif phylo_data.get("ascii_tree"):
                # ASCII fallback
                st.code(phylo_data["ascii_tree"], language=None)

            # Show Newick string for download
            newick = phylo_data.get("newick", "")
            if newick:
                with st.expander("📥 Download Newick format"):
                    st.code(newick, language=None)
                    st.download_button(
                        "⬇️ Download Newick",
                        data=newick,
                        file_name="proteusiq_tree.nwk",
                        mime="text/plain",
                    )
        else:
            st.info(phylo_data.get("message", "Tree not constructed."))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 4: ADVANCED
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_advanced:
        sequence = data.get("sequence", "")

        # ─── Hydropathy Plot ───
        st.markdown("### 📉 Hydropathy Profile (Kyte-Doolittle)")
        hydro = compute_hydropathy(sequence, window=9)
        if hydro:
            hydro_df = pd.DataFrame(hydro, columns=["Position", "Score"]).set_index("Position")
            st.line_chart(hydro_df, height=250)
            st.caption(
                "Positive scores = hydrophobic regions (potential TM helices). "
                "Window size: 9 residues."
            )
        else:
            st.info("Sequence too short for hydropathy analysis.")

        # ─── Sequence Annotation Viewer ───
        st.markdown("### 🧬 Annotated Sequence Viewer")
        entropy_scores = data.get("entropy", {}).get("entropy_scores", {})
        ss_assign = data.get("secondary_structure", {}).get("assignments", {})
        tm_positions = data.get("tm_prediction", {}).get("tm_helix_positions", [])
        conserved_pos = data.get("entropy", {}).get("conserved_positions", [])
        disorder_scores = data.get("disorder", {}).get("scores", {})
        domain_annotations = data.get("domains", [])  # PROSITE domains
        tm_data = data.get("tm_prediction", {})

        seq_html = generate_sequence_annotation_html(
            sequence,
            entropy_scores=entropy_scores,
            ss_assignments=ss_assign,
            tm_positions=tm_positions,
            conserved_positions=conserved_pos,
            domain_annotations=domain_annotations,
            signal_peptide=tm_data,
            disorder_scores=disorder_scores,
        )
        if seq_html:
            components.html(seq_html, height=min(600, 120 + (len(sequence) // 60) * 100), scrolling=True)

        # ─── Disorder Prediction ───
        st.markdown("### 🌀 Disorder Prediction")
        disorder_data = data.get("disorder", {})
        if disorder_data.get("scores"):
            st.markdown(f"**{disorder_data.get('summary', '')}**")

            c1, c2 = st.columns(2)
            c1.metric("Disorder Content", f"{disorder_data.get('disorder_content', 0)}%")
            c2.metric("Disordered Regions", len(disorder_data.get('disordered_regions', [])))

            if disorder_data.get("disordered_regions"):
                regions_str = ", ".join(
                    f"{s}–{e}" for s, e in disorder_data["disordered_regions"]
                )
                st.caption(f"Disordered regions: {regions_str}")

            # Disorder profile as line chart
            dis_scores = disorder_data["scores"]
            if dis_scores:
                dis_data = sorted(dis_scores.items())
                dis_df = pd.DataFrame(dis_data, columns=["Position", "Score"]).set_index("Position")
                st.line_chart(dis_df, height=200)
                st.caption(
                    "Score > 0.5 = predicted disordered. Based on TOP-IDP amino acid propensity scale."
                )
        else:
            st.info(disorder_data.get("summary", "Disorder prediction not available."))

        # ─── Localization Detail ───
        st.markdown("### 📍 Localization Detail")
        tm = data.get("tm_prediction", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Location", tm.get("localization", "Unknown"))
        c2.metric("Signal Peptide", "Yes ✂️" if tm.get("signal_peptide") else "No")
        c3.metric("TM Helices", tm.get("tm_helices", 0))

        if tm.get("cleavage_site"):
            st.info(f"Predicted cleavage site at position ~{tm['cleavage_site']}")
        if tm.get("tm_helix_positions"):
            positions = ", ".join(f"{s+1}–{e}" for s, e in tm["tm_helix_positions"])
            st.caption(f"TM helix positions: {positions}")

        # ─── GO Terms ───
        uni = data.get("uniprot_meta", {})
        if uni.get("found") and uni.get("go_terms"):
            st.markdown("### 🏷️ Gene Ontology Terms")
            go_df = pd.DataFrame(uni["go_terms"])
            st.dataframe(
                go_df.rename(columns={"id": "GO ID", "term": "Term"}),
                use_container_width=True, hide_index=True,
            )

        # ─── Export ───
        st.markdown("---")
        st.markdown("### 📄 Export Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            try:
                html_report = generate_html(data)
                st.download_button(
                    "⬇️ HTML Report",
                    data=html_report,
                    file_name="proteusiq_report.html",
                    mime="text/html",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Report generation failed: {e}")

        with col2:
            # JSON export
            json_safe = _make_json_safe(data)
            json_str = json.dumps(json_safe, indent=2, default=str)
            st.download_button(
                "⬇️ JSON Data",
                data=json_str,
                file_name="proteusiq_results.json",
                mime="application/json",
                use_container_width=True,
            )

        with col3:
            try:
                from report.generate_report import generate_pdf
                pdf_bytes = generate_pdf(data)
                st.download_button(
                    "⬇️ PDF Report",
                    data=pdf_bytes,
                    file_name="proteusiq_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except ImportError:
                st.caption("PDF requires WeasyPrint")
            except Exception as e:
                st.caption(f"PDF failed: {e}")


def _make_json_safe(data: dict) -> dict:
    """Convert data dict to JSON-serializable form."""
    safe = {}
    skip_keys = {"pdb_content", "pdb_path"}  # too large / not useful

    for key, value in data.items():
        if key in skip_keys:
            continue
        try:
            json.dumps(value, default=str)
            safe[key] = value
        except (TypeError, ValueError):
            safe[key] = str(value)

    return safe


def _show_landing_page():
    """Display the landing page when no analysis has been run."""
    st.markdown("""
    ### Welcome! 👋

    **ProteusIQ** is a comprehensive protein Sequence & Structural Analysis platform combining
    evolutionary conservation, structural analysis, and spatial statistics.

    #### ✨ Features

    | Tab | Feature | Description |
    |---|---|---|
    | 📊 **Overview** | Physicochemical | MW, pI, composition, stability, localization |
    | | Inference | Rule-based functional prediction |
    | | Homology | NCBI + EBI BLAST with E-value filtering |
    | | Domains | PROSITE patterns + optional InterPro search |
    | 🏗️ **Structure** | 3D Viewer | 3Dmol.js with 6 coloring modes |
    | | Secondary structure | Helix/sheet/coil from PDB |
    | | Ligand contacts | Quantitative residue-ligand analysis |
    | 📈 **Evolution** | Conservation | Shannon entropy scoring |
    | | Clustering | Permutation test for spatial clusters |
    | | MSA / Tree | Clustal Omega + NJ tree with bootstrap |
    | 🔬 **Advanced** | Hydropathy | Kyte-Doolittle profile plot |
    | | Sequence viewer | Multi-track annotated sequence (SS, conservation, TM, domains, disorder) |
    | | Disorder | TOP-IDP propensity-based disorder prediction |
    | | Export | HTML, JSON, and PDF downloads |

    ---

    **Get started:** Paste a sequence in the sidebar or load an example, then click **Analyze Protein**.
    """)

    st.markdown("---")
    st.caption(
        "ProteusIQ • Streamlit • Biopython • NCBI BLAST • EBI BLAST • "
        "RCSB PDB • AlphaFold • 3Dmol.js • Clustal Omega • InterPro"
    )


if __name__ == "__main__":
    main()
