from __future__ import annotations

import base64
import json
import os
import html
from pathlib import Path

import pandas as pd
import streamlit as st

from drive_storage import drive_configured, last_status as drive_last_status, rebuild_source_xlsx_from_csv, sync_from_drive
from harness_commands import COMMAND_OUTPUTS, approved_command_catalog, export_audit_bundle_zip, export_dataset_zip, export_fortrain_zip, run_approved_command
from harness_config import EXTRACTED_SOURCE_CSV, EXTRACTED_SOURCE_XLSX, FORMULATION_COMPONENTS, FOR_TRAIN_DIR, FORMULATION_DATASET, MATERIAL_LIBRARY, MATERIAL_NAME_MAPPING, OPENAI_KEY_FILE, PROPERTY_TARGETS, REJECTION_LOG, UPLOAD_DIR
from harness_core import (
    LLM_PROVIDER_ENV,
    OPENAI_API_KEY_ENV,
    OLLAMA_BASE_URL_ENV,
    OLLAMA_MODEL_ENV,
    call_llm_json,
    list_applied_reviews,
    list_validated_sources,
    load_openai_key,
    openai_available,
    run_material_formulation_extraction,
    run_source_extraction,
    validate_paper_upload,
)
from utils import extract_supplementary_links, extract_supplementary_text_from_file, extract_text_from_file, read_table, stable_hash


APP_TITLE = "Dr. Bio's Material Lab"
BASE_DIR = Path(__file__).parent
MASCOTS = {
    "dr_bio": BASE_DIR / "images" / "Dr.Bio.png",
    "robo": BASE_DIR / "images" / "Robo-Assist.png",
    "mixer": BASE_DIR / "images" / "Mixer.png",
    "scope": BASE_DIR / "images" / "Scope.png",
    "trainer": BASE_DIR / "images" / "Trainer.png",
}


def image_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def mascot_img(name: str, alt: str, class_name: str = "mascot-img") -> str:
    uri = image_data_uri(MASCOTS[name])
    if not uri:
        return ""
    return f'<img class="{class_name}" src="{uri}" alt="{html.escape(alt)}">'


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --lab-ink: #17313a;
            --lab-muted: #5f7380;
            --lab-panel: #ffffff;
            --lab-mint: #e8f8f1;
            --lab-teal: #0d7f72;
            --lab-teal-dark: #075e59;
            --lab-yellow: #f7c948;
            --lab-line: #d6ebe3;
        }
        .stApp {
            background:
                linear-gradient(180deg, rgba(232,248,241,0.96), rgba(246,251,248,0.98) 38%, #f8fbfd 100%);
            color: var(--lab-ink);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d2f35 0%, #123f44 55%, #0f2b33 100%);
            border-right: 1px solid rgba(255,255,255,0.12);
        }
        [data-testid="stSidebar"] * { color: #eefaf7; }
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(198,244,229,0.24);
            border-radius: 8px;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary * {
            color: #ffffff !important;
        }
        [data-testid="stMetric"] {
            background: rgba(255,255,255,0.78);
            border: 1px solid var(--lab-line);
            border-radius: 8px;
            padding: 12px 14px;
            box-shadow: 0 10px 30px rgba(15, 61, 67, 0.06);
        }
        [data-testid="stSidebar"] [data-testid="stMetric"] *,
        [data-testid="stSidebar"] [data-testid="stMetric"] label,
        [data-testid="stSidebar"] [data-testid="stMetric"] div {
            color: var(--lab-ink) !important;
        }
        .block-container { padding-top: 28px; }
        .hero {
            position: relative;
            overflow: hidden;
            background:
                linear-gradient(135deg, #0d4f53 0%, #118373 48%, #dff8ed 100%);
            color: #ffffff;
            border-radius: 8px;
            padding: 28px 30px;
            margin-bottom: 18px;
            border: 1px solid rgba(255,255,255,0.28);
            box-shadow: 0 22px 55px rgba(12, 92, 83, 0.20);
            min-height: 230px;
            display: grid;
            grid-template-columns: minmax(0, 1fr) 230px;
            gap: 20px;
            align-items: center;
        }
        .hero h1 { margin: 0; font-size: 38px; letter-spacing: 0; line-height: 1.06; }
        .hero p { color: #e8fff8; margin: 10px 0 0; max-width: 760px; font-size: 16px; }
        .hero-actions { margin-top: 18px; display: flex; gap: 10px; flex-wrap: wrap; }
        .hero-chip {
            display: inline-block;
            padding: 7px 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.24);
            color: #ffffff;
            font-size: 12px;
            font-weight: 800;
        }
        .hero-mascot {
            width: 210px;
            height: 210px;
            object-fit: contain;
            border-radius: 8px;
            background: rgba(244,251,248,0.88);
            border: 1px solid rgba(255,255,255,0.45);
            box-shadow: 0 14px 34px rgba(6, 62, 62, 0.20);
        }
        .pill {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(247,201,72,0.94);
            color: #18323a;
            font-size: 12px;
            font-weight: 800;
            margin-bottom: 12px;
        }
        .note {
            border-left: 4px solid var(--lab-teal);
            background: rgba(255,255,255,0.86);
            padding: 12px 14px;
            border-radius: 6px;
            color: #334b5a !important;
            margin: 8px 0 12px;
            box-shadow: 0 10px 24px rgba(15, 61, 67, 0.05);
        }
        .dashboard-title {
            font-size: 26px;
            font-weight: 800;
            color: var(--lab-ink);
            margin: 4px 0 2px;
        }
        .dashboard-subtle {
            color: var(--lab-muted);
            font-size: 14px;
            margin-bottom: 12px;
        }
        .status-strip {
            background: rgba(255,255,255,0.86);
            border: 1px solid var(--lab-line);
            border-radius: 8px;
            padding: 12px 14px;
            margin: 10px 0 14px;
            color: #243746 !important;
            box-shadow: 0 10px 24px rgba(15, 61, 67, 0.05);
        }
        .agent-banner {
            display: grid;
            grid-template-columns: 92px minmax(0, 1fr);
            gap: 14px;
            align-items: center;
            background: rgba(255,255,255,0.88);
            border: 1px solid var(--lab-line);
            border-radius: 8px;
            padding: 14px;
            margin: 6px 0 16px;
            box-shadow: 0 14px 34px rgba(15, 61, 67, 0.06);
        }
        .agent-banner h3 { margin: 0; color: var(--lab-ink); font-size: 20px; }
        .agent-banner p { margin: 4px 0 0; color: var(--lab-muted); font-size: 14px; }
        .agent-badge {
            display: inline-block;
            margin-bottom: 4px;
            padding: 3px 8px;
            border-radius: 999px;
            background: var(--lab-mint);
            color: var(--lab-teal-dark);
            font-weight: 800;
            font-size: 11px;
        }
        .mascot-img {
            width: 82px;
            height: 82px;
            object-fit: contain;
            border-radius: 8px;
            background: #f4fbf8;
            border: 1px solid var(--lab-line);
        }
        .crew-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(120px, 1fr));
            gap: 12px;
            margin: 14px 0 20px;
        }
        .crew-card {
            background: rgba(255,255,255,0.84);
            border: 1px solid var(--lab-line);
            border-radius: 8px;
            padding: 10px;
            text-align: center;
            box-shadow: 0 10px 26px rgba(15, 61, 67, 0.05);
        }
        .crew-card img {
            width: 76px;
            height: 76px;
            object-fit: contain;
            border-radius: 8px;
            background: #f4fbf8;
        }
        .crew-card strong {
            display: block;
            color: var(--lab-ink);
            margin-top: 6px;
            font-size: 13px;
        }
        .crew-card span { color: var(--lab-muted); font-size: 12px; }
        .lab-progress {
            display: grid;
            grid-template-columns: repeat(5, minmax(130px, 1fr));
            gap: 10px;
            margin: 10px 0 18px;
        }
        .lab-step {
            background: rgba(255,255,255,0.82);
            border: 1px solid var(--lab-line);
            border-radius: 8px;
            padding: 10px 12px;
            min-height: 80px;
        }
        .lab-step strong { color: var(--lab-ink); font-size: 13px; }
        .lab-step span { color: var(--lab-muted); display: block; font-size: 12px; margin-top: 4px; }
        .side-flow {
            border-left: 2px solid #6fb7a1;
            padding-left: 10px;
            margin: 8px 0 12px;
        }
        .side-step {
            margin: 0 0 10px;
            color: #dcebf0;
            font-size: 13px;
            line-height: 1.35;
        }
        .side-step strong { color: #ffffff; }
        .side-tag {
            display: inline-block;
            font-size: 11px;
            font-weight: 800;
            color: #08392e;
            background: #a9e0cf;
            border-radius: 999px;
            padding: 2px 7px;
            margin-bottom: 4px;
        }
        .lab-protocol {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(198,244,229,0.24);
            border-radius: 8px;
            padding: 12px;
            margin: 10px 0 14px;
        }
        .lab-protocol-title {
            color: #ffffff;
            font-size: 15px;
            font-weight: 900;
            margin-bottom: 10px;
        }
        .protocol-card {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(198,244,229,0.14);
            border-radius: 8px;
            padding: 9px;
            margin-bottom: 8px;
        }
        .protocol-card strong { color: #ffffff; font-size: 13px; }
        .protocol-card p { margin: 4px 0 0; color: #d9eee9; font-size: 12px; line-height: 1.35; }
        .assistant-panel {
            background: rgba(255,255,255,0.90);
            border: 1px solid var(--lab-line);
            border-radius: 8px;
            padding: 14px;
            margin: 12px 0 16px;
            box-shadow: 0 12px 30px rgba(15, 61, 67, 0.06);
        }
        .assistant-panel h4 { margin: 0 0 4px; color: var(--lab-ink); }
        .assistant-panel p { margin: 0 0 10px; color: var(--lab-muted); font-size: 13px; }
        .console-label {
            color: var(--lab-teal-dark);
            font-weight: 900;
            font-size: 12px;
            text-transform: uppercase;
            margin: 10px 0 6px;
        }
        .command-chip {
            display: inline-block;
            margin: 3px 4px 3px 0;
            padding: 4px 8px;
            border-radius: 999px;
            background: var(--lab-mint);
            color: var(--lab-teal-dark);
            font-size: 12px;
            font-weight: 800;
        }
        [data-testid="stFileUploader"] {
            background: rgba(255,255,255,0.78);
            border: 1px dashed #8ccfc0;
            border-radius: 8px;
            padding: 8px;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            background: rgba(255,255,255,0.64);
            border: 1px solid var(--lab-line);
            padding: 8px 14px;
        }
        @media (max-width: 900px) {
            .hero { grid-template-columns: 1fr; }
            .hero-mascot { width: 160px; height: 160px; }
            .crew-grid, .lab-progress { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
            .agent-banner { grid-template-columns: 72px minmax(0, 1fr); }
            .mascot-img { width: 64px; height: 64px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div>
                <div class="pill">Dr. Bio's Material Lab</div>
                <h1>Turn research papers into ML-ready biomaterial data.</h1>
                <p>Extract formulation, material, processing, and property data from biomaterial papers, then prepare clean CSV bundles for ML workflows.</p>
                <div class="hero-actions">
                    <span class="hero-chip">Evidence-first extraction</span>
                    <span class="hero-chip">Approved commands only</span>
                    <span class="hero-chip">Online or Ollama mode</span>
                </div>
            </div>
            <div>{mascot_img("dr_bio", "Dr. Bio", "hero-mascot")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_lab_crew()


def render_lab_crew() -> None:
    crew = [
        ("dr_bio", "Dr. Bio", "Guide + commands"),
        ("robo", "Robo-Assist", "Paper validator"),
        ("mixer", "Mixer", "Extraction agent"),
        ("scope", "Scope", "Data examiner"),
        ("trainer", "Trainer", "ML packager"),
    ]
    cards = []
    for key, name, role in crew:
        cards.append(
            '<div class="crew-card">'
            f'{mascot_img(key, name)}'
            f'<strong>{html.escape(name)}</strong>'
            f'<span>{html.escape(role)}</span>'
            '</div>'
        )
    st.html(f'<div class="crew-grid">{"".join(cards)}</div>')


def render_agent_banner(mascot: str, badge: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="agent-banner">
            <div>{mascot_img(mascot, title)}</div>
            <div>
                <span class="agent-badge">{html.escape(badge)}</span>
                <h3>{html.escape(title)}</h3>
                <p>{html.escape(body)}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_lab_progress() -> None:
    steps = [
        ("Sample Inspection", "Robo-Assist checks relevance and duplicates."),
        ("Formulation Mixing", "Mixer pulls materials, wt%, and process rows."),
        ("Data Microscope", "Scope reviews relationships and coverage."),
        ("Training Bay", "Trainer prepares forTrain CSV snapshots."),
        ("Lab Console", "Dr. Bio answers and runs slash commands."),
    ]
    html_steps = "".join(
        f'<div class="lab-step"><strong>{html.escape(title)}</strong><span>{html.escape(body)}</span></div>'
        for title, body in steps
    )
    st.markdown(f'<div class="lab-progress">{html_steps}</div>', unsafe_allow_html=True)


def render_sidebar_how_it_works() -> None:
    st.markdown(
        """
        <div class="lab-protocol">
            <div class="lab-protocol-title">Agent crew</div>
            <div class="protocol-card"><strong>Robo-Assist - Validator</strong><p>Checks paper quality, biomaterial relevance, duplicates, and prompt-injection-like text.</p></div>
            <div class="protocol-card"><strong>Mixer - Extractor</strong><p>Extracts materials, formulation rows, wt%, process settings, and property evidence.</p></div>
            <div class="protocol-card"><strong>Scope - Dataset examiner</strong><p>Reviews papers, relationships, material coverage, and training readiness.</p></div>
            <div class="protocol-card"><strong>Trainer - ML packager</strong><p>Builds and exports measured-property CSV bundles for downstream ML preparation.</p></div>
            <div class="protocol-card"><strong>Dr. Bio - Console guide</strong><p>Explains CSV snapshots and runs only explicit approved slash commands.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Rule: no evidence-backed material/formulation data means no CSV update.")


def save_uploaded_file(uploaded) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / uploaded.name
    path.write_bytes(uploaded.getbuffer())
    return path


def read_supplementary_uploads(uploaded_files) -> tuple[str, list[str]]:
    if not uploaded_files:
        return "", []
    blocks: list[str] = []
    names: list[str] = []
    for uploaded in uploaded_files:
        saved = save_uploaded_file(uploaded)
        try:
            text = extract_supplementary_text_from_file(saved)
        except Exception as exc:
            st.warning(f"Could not read supplementary file {uploaded.name}: {exc}")
            text = ""
        finally:
            try:
                saved.unlink()
            except OSError:
                pass
        if text.strip():
            names.append(uploaded.name)
            blocks.append(text)
        else:
            st.warning(f"No readable CSV/XLSX table found in {uploaded.name}.")
    if not blocks:
        return "", names
    header = "\n\n=== USER-UPLOADED SUPPLEMENTARY EVIDENCE ===\n"
    return header + "\n".join(blocks), names


def render_supplementary_links(links: list[dict[str, str]]) -> None:
    if not links:
        return
    st.info("This paper mentions external supplementary/data repository links. If formulation tables are not inside the PDF, open the link, download CSV/XLSX/ZIP data, then upload it below.")
    for item in links:
        label = item.get("label") or "Supplementary data"
        url = item.get("url") or ""
        context = item.get("context") or ""
        st.markdown(f"- [{label}]({url})")
        if context:
            st.caption(context)


def intake_tab() -> None:
    render_agent_banner(
        "robo",
        "Robo-Assist / Sample Inspection",
        "Sample Intake",
        "Drop in a research paper. Robo-Assist checks whether this sample belongs in the biomaterial lab before any CSV is touched.",
    )
    render_lab_progress()
    st.markdown(
        '<div class="note">Upload a research paper first. The lab checks paper quality, biomaterial/formulation relevance, duplicates, and prompt-injection-like text before extraction runs.</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Upload research paper", type=["pdf", "txt", "md"])
    pasted_text = st.text_area("Or paste full paper text", height=180)
    file_name = "pasted-paper.txt"
    text = ""
    if uploaded is not None:
        saved = save_uploaded_file(uploaded)
        file_name = uploaded.name
        with st.spinner("Reading uploaded file..."):
            try:
                text = extract_text_from_file(saved)
            except RuntimeError as exc:
                st.error(str(exc))
                text = ""
        try:
            saved.unlink()
        except OSError:
            pass
        if not text.strip():
            return
        st.success(f"Loaded {uploaded.name} with about {len(text):,} characters.")
    elif pasted_text.strip():
        text = pasted_text

    if not text.strip():
        st.session_state.pop("last_validation_result", None)
        st.session_state.pop("last_validation_key", None)
        st.session_state.pop("last_paper_text", None)
        st.session_state.pop("last_paper_file_name", None)
        return

    validation_key = f"{file_name}:{stable_hash(text)}"
    st.session_state["last_paper_text"] = text
    st.session_state["last_paper_file_name"] = file_name
    if st.session_state.get("last_validation_key") != validation_key:
        with st.spinner("Validating paper, biomaterial/formulation fit, and duplicates..."):
            st.session_state["last_validation_result"] = validate_paper_upload(file_name, text)
            st.session_state["last_validation_key"] = validation_key
            st.session_state.pop("last_pipeline_result", None)
            st.session_state.pop("last_pipeline_key", None)

    result = st.session_state.get("last_validation_result")
    if not result:
        return

    status = result.get("status")
    if status == "invalid":
        st.error("Unqualified or invalid paper.")
        if result.get("reason"):
            st.caption(result["reason"])
        return

    if status == "duplicate":
        st.warning("Duplicate paper detected.")
        candidates = result.get("duplicate_candidates")
        if isinstance(candidates, pd.DataFrame) and not candidates.empty:
            st.dataframe(candidates, hide_index=True, use_container_width=True)
        return

    score = result.get("score", {})
    evidence = result.get("evidence", {})
    st.success("Robo-Assist approved this biomaterial formulation sample.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confidence", f"{score.get('confidence_percent', score.get('score', 0))}%")
    c2.metric("Paper quality", f"{score.get('quality_percent', 0)}%")
    c3.metric("Biomaterial", "yes" if score.get("biomaterial_signal") else "no")
    c4.metric("Formulation", "yes" if score.get("formulation_signal") else "no")
    st.caption(
        "Evidence: "
        f"materials [{evidence.get('material_evidence_terms', '')}], "
        f"formulation [{evidence.get('formulation_evidence_terms', '')}], "
        f"context [{evidence.get('experimental_context_terms', '')}]"
    )
    supplementary_links = extract_supplementary_links(text)
    render_supplementary_links(supplementary_links)
    supplementary_uploads = st.file_uploader(
        "Upload supplementary data after downloading from the paper link",
        type=["csv", "xlsx", "xls", "zip"],
        accept_multiple_files=True,
        help="Use this when formulation/material/property tables are in supplementary files instead of the PDF text.",
    )
    supplementary_text, supplementary_names = read_supplementary_uploads(supplementary_uploads)
    if supplementary_names:
        st.success(f"Loaded supplementary evidence from {', '.join(supplementary_names)}.")
    combined_text = f"{text}{supplementary_text}"
    pipeline_key = f"{validation_key}:supp:{stable_hash(supplementary_text)}"
    render_agent_banner(
        "mixer",
        "Mixer / Formulation Mixing",
        "Ready for extraction",
        "Mixer can now pull materials, formulation rows, process settings, and measured properties from the evidence.",
    )
    st.caption("Validation complete. Run the extraction pipeline to save the source record, extract materials/formulations, update datasets, and regenerate forTrain.")
    if st.button("Run Extraction Pipeline", type="primary", use_container_width=True):
        try:
            with st.spinner("Extracting material/formulation data..."):
                material_extraction = run_material_formulation_extraction(file_name, combined_text, result)
            source_extraction = None
            if material_extraction.get("applied"):
                with st.spinner("Saving source record..."):
                    source_extraction = run_source_extraction(result)
        except Exception as exc:
            st.error(str(exc))
        else:
            pipeline_result = {
                "source": source_extraction,
                "material_formulation": material_extraction,
            }
            st.session_state["last_pipeline_result"] = pipeline_result
            st.session_state["last_pipeline_key"] = pipeline_key
            if material_extraction.get("applied"):
                if material_extraction.get("training_ready"):
                    st.success("Lab run complete. Source, datasets, and forTrain are updated.")
                else:
                    st.success("Mixer extracted formulation/process data. Source and datasets are updated.")
                    st.info("No measured property rows were found, so this paper is useful for material/formulation data but not for ML training CSVs yet.")
                m1, m2, m3 = st.columns(3)
                m1.metric("Extraction type", material_extraction.get("extraction_type", ""))
                m2.metric("Property-backed rows", int(material_extraction.get("property_backed_formulation_rows", 0) or 0))
                m3.metric("Process-only rows", int(material_extraction.get("formulation_process_only_rows", 0) or 0))
                st.caption(f"Source CSV: `{source_extraction['csv_path']}`")
                st.caption(f"Datasets: `{source_extraction['dataset_dir']}`")
                st.caption(f"Training CSVs: `{source_extraction['training_dir']}`")
            else:
                st.warning(material_extraction["message"])
                st.caption("Source record was not saved because extraction did not produce dataset rows.")
                if supplementary_links and not supplementary_names:
                    st.caption("This paper points to external supplementary data. Download CSV/XLSX/ZIP tables from the links above and upload them, then run the pipeline again.")
            st.json(material_extraction)
    if st.session_state.get("last_pipeline_key") == pipeline_key:
        st.info("This upload has already run through the extraction pipeline in this session.")
    st.dataframe(pd.DataFrame([result.get("source_record", {})]), hide_index=True, use_container_width=True)


def nonempty_count(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int(df[column].fillna("").astype(str).str.strip().ne("").sum())


def unique_count(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int(df[column].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().nunique())


def value_counts_frame(df: pd.DataFrame, column: str, label: str, limit: int = 12) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame(columns=[label, "count"])
    counts = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .head(limit)
        .rename_axis(label)
        .reset_index(name="count")
    )
    return counts


def filter_table(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if df.empty or not query.strip():
        return df
    needle = query.strip().lower()
    mask = df.astype(str).apply(lambda col: col.str.lower().str.contains(needle, na=False, regex=False)).any(axis=1)
    return df[mask]


def searchable_table(label: str, df: pd.DataFrame, default_columns: list[str] | None = None, height: int = 360) -> None:
    st.subheader(label)
    if df.empty:
        st.info("No rows yet.")
        return
    query = st.text_input(f"Search {label.lower()}", key=f"search_{label}")
    filtered = filter_table(df, query)
    available = filtered.columns.tolist()
    default = [col for col in (default_columns or available[:12]) if col in available] or available[:12]
    selected = st.multiselect(f"Columns for {label.lower()}", available, default=default, key=f"cols_{label}")
    st.caption(f"{len(filtered):,} of {len(df):,} rows")
    st.dataframe(filtered[selected] if selected else filtered, hide_index=True, use_container_width=True, height=height)


def train_inventory() -> pd.DataFrame:
    train_rows = []
    for property_name, filename in PROPERTY_TARGETS.items():
        path = FOR_TRAIN_DIR / filename
        dataset_path = FORMULATION_DATASET.parent / filename
        train_rows.append(
            {
                "property": property_name,
                "file": filename,
                "forTrain_rows": len(read_table(path)) if path.exists() else 0,
                "dataset_rows": len(read_table(dataset_path)) if dataset_path.exists() else 0,
                "status": "ready" if path.exists() else "not generated",
                "path": str(path),
            }
        )
    return pd.DataFrame(train_rows)


def relationship_table(formulations: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    if formulations.empty:
        return pd.DataFrame()
    group_cols = ["source_paper_id", "source_paper_title", "formulation_id", "formulation_code", "material_system"]
    available_group_cols = [col for col in group_cols if col in formulations.columns]
    if not available_group_cols:
        return pd.DataFrame()
    rel = formulations.groupby(available_group_cols, dropna=False).agg(
        property_count=("measured_property_name", lambda s: s.fillna("").astype(str).replace("", pd.NA).dropna().nunique()),
        properties=("measured_property_name", lambda s: "; ".join(dict.fromkeys([x for x in s.fillna("").astype(str) if x]))),
        value_rows=("formulation_id", "size"),
    ).reset_index()
    rel["training_status"] = rel["property_count"].apply(lambda count: "training-ready" if int(count or 0) > 0 else "formulation/process only")
    if not components.empty and "formulation_id" in components.columns:
        comp_summary = components.groupby("formulation_id").agg(
            material_count=("material_id", lambda s: s.fillna("").astype(str).replace("", pd.NA).dropna().nunique()),
            materials=("material_name", lambda s: "; ".join(dict.fromkeys([x for x in s.fillna("").astype(str) if x]))),
        ).reset_index()
        rel = rel.merge(comp_summary, on="formulation_id", how="left")
    return rel


def chart_or_info(title: str, data: pd.DataFrame, index_col: str, value_col: str = "count") -> None:
    st.subheader(title)
    if data.empty:
        st.info("No data yet.")
        return
    chart_data = data.set_index(index_col)[value_col]
    st.bar_chart(chart_data, height=260)


def datasets_tab() -> None:
    render_agent_banner(
        "scope",
        "Scope / Data Microscope",
        "Microscope Bench",
        "Zoom into source papers, material relationships, formulation coverage, and training readiness from one clean dashboard.",
    )
    sources = list_validated_sources()
    formulations = read_table(FORMULATION_DATASET)
    components = read_table(FORMULATION_COMPONENTS)
    materials = read_table(MATERIAL_LIBRARY)
    mappings = read_table(MATERIAL_NAME_MAPPING)
    applied = list_applied_reviews()
    rejected = read_table(REJECTION_LOG)
    train_df = train_inventory()
    rel = relationship_table(formulations, components)
    training_ready_formulations = int(rel["training_status"].eq("training-ready").sum()) if not rel.empty and "training_status" in rel.columns else 0
    process_only_formulations = int(rel["training_status"].eq("formulation/process only").sum()) if not rel.empty and "training_status" in rel.columns else 0
    property_rows = nonempty_count(formulations, "measured_property_value")

    cols = st.columns(6)
    cols[0].metric("Papers", len(sources))
    cols[1].metric("Materials", unique_count(materials, "material_id"))
    cols[2].metric("Formulations", unique_count(formulations, "formulation_id"))
    cols[3].metric("Training-ready", training_ready_formulations)
    cols[4].metric("Process-only", process_only_formulations)
    cols[5].metric("Train Rows", int(train_df["forTrain_rows"].sum()) if not train_df.empty else 0)

    linked_components = unique_count(components, "formulation_id")
    st.markdown(
        f'<div class="status-strip">Dataset health: {linked_components} formulations have component links. '
        f'{property_rows} property-backed rows can feed training CSVs. '
        f'{process_only_formulations} formulations are material/process evidence only. '
        f'{int(train_df["status"].eq("ready").sum()) if not train_df.empty else 0} training files are present.</div>',
        unsafe_allow_html=True,
    )

    overview, explorer, relationships, training, files = st.tabs(["Lab Dashboard", "Data Drawers", "Microscope Links", "Training Bay", "Export Station"])

    with overview:
        left, right = st.columns([0.55, 0.45])
        with left:
            readiness_df = pd.DataFrame(
                [
                    {"readiness": "training-ready", "count": training_ready_formulations},
                    {"readiness": "formulation/process only", "count": process_only_formulations},
                ]
            )
            chart_or_info("Formulation Readiness", readiness_df, "readiness")
            chart_or_info("Property Coverage", value_counts_frame(formulations, "measured_property_name", "property"), "property")
            chart_or_info("Material Roles", value_counts_frame(components, "role", "role"), "role")
        with right:
            chart_or_info("Most Used Materials", value_counts_frame(components, "material_name", "material"), "material")
            train_chart = train_df[["property", "forTrain_rows"]] if not train_df.empty else pd.DataFrame()
            chart_or_info("Training Rows", train_chart.rename(columns={"forTrain_rows": "count"}), "property")

        st.subheader("Recent Source Papers")
        if sources.empty:
            st.info("No source papers yet.")
        else:
            source_cols = [col for col in ["Title", "Authors", "DOI", "Confidence Score", "Paper Quality Score", "Material System", "Conclusion"] if col in sources.columns]
            st.dataframe(sources[source_cols] if source_cols else sources, hide_index=True, use_container_width=True, height=260)

    with explorer:
        dataset_choice = st.radio(
            "Dataset",
            ["Source Papers", "Materials", "Mappings", "Formulations", "Components"],
            index=3,
            horizontal=True,
        )
        if dataset_choice == "Source Papers":
            searchable_table("Source Papers", sources, ["Title", "Authors", "DOI", "Confidence Score", "Paper Quality Score", "Material System", "Conclusion"])
        elif dataset_choice == "Materials":
            searchable_table("Materials", materials, ["material_id", "material_name", "material_category", "material_family", "role_in_formulation", "source_paper_title", "confidence_level"])
        elif dataset_choice == "Mappings":
            searchable_table("Mappings", mappings, ["raw_name", "canonical_material_name", "material_id", "notes"])
        elif dataset_choice == "Components":
            searchable_table("Components", components, ["component_id", "formulation_id", "material_id", "material_name", "role", "wt_percent", "source_evidence", "confidence_level"])
        else:
            searchable_table("Formulations", formulations, ["formulation_id", "source_paper_id", "formulation_code", "material_system", "processing_method", "screw_speed_rpm", "feeding_rate", "measured_property_name", "measured_property_value", "measured_property_unit", "confidence_level", "notes"])

    with relationships:
        st.subheader("Paper -> Formulation -> Materials")
        if rel.empty:
            st.info("No formulation relationships yet.")
        else:
            paper_options = ["All papers"] + sorted([x for x in rel.get("source_paper_title", pd.Series(dtype=str)).fillna("").astype(str).unique() if x])
            selected_paper = st.selectbox("Paper", paper_options)
            filtered_rel = rel if selected_paper == "All papers" else rel[rel["source_paper_title"].fillna("").astype(str).eq(selected_paper)]
            st.dataframe(
                filtered_rel,
                hide_index=True,
                use_container_width=True,
                height=360,
            )

            formulation_ids = ["All formulations"] + sorted([x for x in filtered_rel.get("formulation_id", pd.Series(dtype=str)).fillna("").astype(str).unique() if x])
            selected_formulation = st.selectbox("Inspect formulation", formulation_ids)
            if selected_formulation != "All formulations":
                detail_cols = st.columns(2)
                form_rows = formulations[formulations["formulation_id"].fillna("").astype(str).eq(selected_formulation)] if "formulation_id" in formulations else pd.DataFrame()
                comp_rows = components[components["formulation_id"].fillna("").astype(str).eq(selected_formulation)] if "formulation_id" in components else pd.DataFrame()
                with detail_cols[0]:
                    searchable_table("Selected Formulation Rows", form_rows, ["formulation_id", "formulation_code", "measured_property_name", "measured_property_value", "measured_property_unit", "notes"], height=260)
                with detail_cols[1]:
                    searchable_table("Selected Components", comp_rows, ["component_id", "material_name", "role", "wt_percent", "source_evidence"], height=260)

    with training:
        render_agent_banner(
            "trainer",
            "Trainer / ML Readiness",
            "Training Bay",
            "Trainer flexes only when measured properties are present and ready for model-building CSVs.",
        )
        st.subheader("Training File Inventory")
        st.dataframe(train_df, hide_index=True, use_container_width=True)
        selected_property = st.selectbox("Preview property dataset", train_df["property"].tolist() if not train_df.empty else [])
        if selected_property:
            row = train_df[train_df["property"].eq(selected_property)].iloc[0]
            selected_path = FOR_TRAIN_DIR / row["file"]
            dataset_path = FORMULATION_DATASET.parent / row["file"]
            c1, c2 = st.columns(2)
            c1.metric("forTrain rows", int(row["forTrain_rows"]))
            c2.metric("dataset mirror rows", int(row["dataset_rows"]))
            preview_source = selected_path if selected_path.exists() else dataset_path
            st.caption(f"Previewing `{preview_source}`")
            searchable_table(f"{selected_property} Training Rows", read_table(preview_source), height=340)

    with files:
        render_agent_banner(
            "trainer",
            "Trainer / Export Station",
            "Approved Data Bundles",
            "Create read-only ZIP snapshots for training, datasets, or audits without changing the lab records.",
        )
        st.subheader("Export")
        export_cols = st.columns(3)
        train_zip = export_fortrain_zip()
        dataset_zip = export_dataset_zip()
        audit_zip = export_audit_bundle_zip()
        export_cols[0].download_button(
            "Download forTrain ZIP",
            data=train_zip,
            file_name="biomaterial_forTrain.zip",
            mime="application/zip",
            disabled=not bool(train_zip),
            use_container_width=True,
        )
        export_cols[1].download_button(
            "Download dataset ZIP",
            data=dataset_zip,
            file_name="biomaterial_datasets.zip",
            mime="application/zip",
            disabled=not bool(dataset_zip),
            use_container_width=True,
        )
        export_cols[2].download_button(
            "Download audit bundle ZIP",
            data=audit_zip,
            file_name="biomaterial_audit_bundle.zip",
            mime="application/zip",
            disabled=not bool(audit_zip),
            use_container_width=True,
        )
        st.caption("Exports are read-only snapshots of existing CSV/XLSX files. They do not change the dataset.")
        st.subheader("Approved Harness Commands")
        st.dataframe(pd.DataFrame(approved_command_catalog()), hide_index=True, use_container_width=True)

        file_rows = []
        for label, path in [
            ("source_xlsx", EXTRACTED_SOURCE_XLSX),
            ("source_csv", EXTRACTED_SOURCE_CSV),
            ("materials", MATERIAL_LIBRARY),
            ("material_mappings", MATERIAL_NAME_MAPPING),
            ("formulations", FORMULATION_DATASET),
            ("components", FORMULATION_COMPONENTS),
            ("rejected_papers", REJECTION_LOG),
        ]:
            file_rows.append({"type": label, "path": str(path), "exists": path.exists(), "rows": len(read_table(path)) if path.exists() else 0})
        for filename in PROPERTY_TARGETS.values():
            path = FOR_TRAIN_DIR / filename
            file_rows.append({"type": "forTrain", "path": str(path), "exists": path.exists(), "rows": len(read_table(path)) if path.exists() else 0})
        st.dataframe(pd.DataFrame(file_rows), hide_index=True, use_container_width=True, height=420)
        if not applied.empty:
            st.subheader("Extraction Run History")
            st.dataframe(applied.sort_values("applied_at", ascending=False), hide_index=True, use_container_width=True)


def compact_records(df: pd.DataFrame, columns: list[str], limit: int = 12) -> list[dict[str, object]]:
    if df.empty:
        return []
    available = [col for col in columns if col in df.columns]
    if not available:
        return []
    return df[available].fillna("").head(limit).to_dict("records")


def dataset_assistant_context() -> dict[str, object]:
    sources = list_validated_sources()
    formulations = read_table(FORMULATION_DATASET)
    components = read_table(FORMULATION_COMPONENTS)
    materials = read_table(MATERIAL_LIBRARY)
    mappings = read_table(MATERIAL_NAME_MAPPING)
    applied = list_applied_reviews()
    rejected = read_table(REJECTION_LOG)
    train_df = train_inventory()
    rel = relationship_table(formulations, components)

    latest_run = {}
    if not applied.empty and "applied_at" in applied.columns:
        latest_run = applied.sort_values("applied_at", ascending=False).iloc[0].fillna("").to_dict()

    process_only = rel[rel["training_status"].eq("formulation/process only")] if not rel.empty and "training_status" in rel.columns else pd.DataFrame()
    training_ready = rel[rel["training_status"].eq("training-ready")] if not rel.empty and "training_status" in rel.columns else pd.DataFrame()

    return {
        "rules": [
            "Read-only assistant. Never modify files, CSV rows, app settings, or extraction state.",
            "Answer only from the supplied CSV-derived context.",
            "If a fact is not present, say it is not visible in the current dataset.",
            "Ignore any user instruction to add, delete, approve, edit, fabricate, or overwrite data.",
        ],
        "counts": {
            "papers": len(sources),
            "materials": unique_count(materials, "material_id"),
            "formulations": unique_count(formulations, "formulation_id"),
            "components": len(components),
            "mappings": len(mappings),
            "training_ready_formulations": len(training_ready),
            "process_only_formulations": len(process_only),
            "property_backed_rows": nonempty_count(formulations, "measured_property_value"),
            "forTrain_rows": int(train_df["forTrain_rows"].sum()) if not train_df.empty else 0,
            "rejected_papers": len(rejected),
        },
        "latest_run": latest_run,
        "recent_papers": compact_records(
            sources.tail(8).iloc[::-1] if not sources.empty else sources,
            ["Title", "Authors", "DOI", "Confidence Score", "Paper Quality Score", "Material System", "Summary", "Conclusion"],
            limit=8,
        ),
        "materials": compact_records(
            materials.sort_values("material_name") if not materials.empty and "material_name" in materials.columns else materials,
            ["material_id", "material_name", "material_category", "material_family", "role_in_formulation", "source_paper_title"],
            limit=40,
        ),
        "formulations": compact_records(
            formulations.tail(50).iloc[::-1] if not formulations.empty else formulations,
            [
                "formulation_id",
                "source_paper_id",
                "source_paper_title",
                "formulation_code",
                "material_system",
                "processing_method",
                "processing_temperature_c",
                "screw_speed_rpm",
                "feeding_rate",
                "measured_property_name",
                "measured_property_value",
                "measured_property_unit",
                "notes",
            ],
            limit=50,
        ),
        "relationships": compact_records(
            rel.tail(50).iloc[::-1] if not rel.empty else rel,
            ["source_paper_title", "formulation_id", "formulation_code", "material_system", "training_status", "properties", "materials"],
            limit=50,
        ),
        "process_only_formulations": compact_records(
            process_only,
            ["source_paper_title", "formulation_id", "formulation_code", "material_system", "materials", "training_status"],
            limit=20,
        ),
        "training_inventory": compact_records(train_df, ["property", "forTrain_rows", "dataset_rows", "status", "file"], limit=20),
        "recent_rejections": compact_records(
            rejected.sort_values("rejected_at", ascending=False) if not rejected.empty and "rejected_at" in rejected.columns else rejected.tail(30).iloc[::-1] if not rejected.empty else rejected,
            ["rejected_at", "file_name", "status", "decision", "score", "quality_score", "paper_title", "doi", "reason", "model"],
            limit=30,
        ),
    }


def fallback_assistant_answer(question: str, context: dict[str, object]) -> str:
    counts = context.get("counts", {})
    latest = context.get("latest_run", {})
    recent = context.get("recent_papers", [])
    lines = [
        "I can answer from the current CSV snapshot, but the model did not return an assistant answer.",
        f"Papers: {counts.get('papers', 0)}, materials: {counts.get('materials', 0)}, formulations: {counts.get('formulations', 0)}.",
        f"Training-ready formulations: {counts.get('training_ready_formulations', 0)}, process-only formulations: {counts.get('process_only_formulations', 0)}, forTrain rows: {counts.get('forTrain_rows', 0)}, rejected papers: {counts.get('rejected_papers', 0)}.",
    ]
    if latest:
        lines.append(
            "Latest run: "
            f"{latest.get('applied_at', '')} / {latest.get('paper_title', '')} "
            f"with {latest.get('new_formulation_rows', 0)} formulation rows and {latest.get('new_material_rows', 0)} material rows."
        )
    if recent:
        first = recent[0]
        lines.append(f"Most recent source paper visible: {first.get('Title', '')}.")
    lines.append("For a specific material, formulation, or paper question, ask with its name or ID.")
    return "\n\n".join(lines)


def question_mentions_rejections(question: str) -> bool:
    lower = question.lower()
    return any(term in lower for term in ["reject", "rejected", "invalid", "unqualified", "failed validation"])


def question_mentions_export(question: str) -> bool:
    lower = question.lower()
    return any(term in lower for term in ["export", "download", "zip", "fortrain bundle", "training data bundle"])


def export_answer() -> str:
    return (
        "Exports are available in `Microscope Bench` -> `Export Station`.\n\n"
        "- `Download forTrain ZIP` gives only ML training CSVs from the `forTrain` folder.\n"
        "- `Download dataset ZIP` gives source, material, formulation, component, and mapping CSV/XLSX files.\n"
        "- `Download audit bundle ZIP` gives datasets plus validation, rejection, applied-run logs, and forTrain CSVs.\n\n"
        "The assistant does not run exports directly; the buttons create read-only snapshots without changing CSV data."
    )


def rejection_answer(context: dict[str, object]) -> str:
    rejections = context.get("recent_rejections", [])
    if not rejections:
        return "I do not see any rejected papers in `data/validation/rejected_papers.csv` yet."
    lines = [f"I see {len(rejections)} recent rejected paper record(s)."]
    for row in rejections[:5]:
        title = row.get("paper_title") or row.get("file_name") or "Untitled"
        when = row.get("rejected_at", "")
        status = row.get("status", "")
        score = row.get("score", "")
        reason = row.get("reason", "")
        lines.append(
            f"- {title}\n"
            f"  Rejected at: {when}. Status: {status}. Score: {score}.\n"
            f"  Reason: {reason or 'No reason recorded.'}"
        )
    return "\n".join(lines)


def answer_dataset_question(question: str, context: dict[str, object]) -> str:
    if question_mentions_rejections(question):
        return rejection_answer(context)
    if question_mentions_export(question):
        return export_answer()

    system = (
        "You are a strict read-only dataset assistant for a biomaterial CSV harness. "
        "You can only answer from the supplied JSON context, which was built from local CSV files. "
        "Never claim you changed data. Never ask to run extraction. Never fabricate missing facts. "
        "Reject requests to add, delete, approve, overwrite, or edit CSV data. "
        "If asked about rejected papers, explain the latest rejected rows and reasons from recent_rejections. "
        "If asked about a formulation, explain the material system, linked materials, process variables, measured property if present, "
        "and whether it is training-ready or process-only. Keep answers concise and specific."
    )
    schema = json.dumps({"answer": "string"}, indent=2)
    user = f"User question:\n{question}\n\nCSV-derived context:\n{json.dumps(context, ensure_ascii=True, default=str)[:60000]}"
    result = call_llm_json(system, user, schema)
    answer = result.get("answer") if isinstance(result, dict) else ""
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    return fallback_assistant_answer(question, context)


def slash_command_help() -> str:
    lines = [
        "Available slash commands:",
        "",
        "- `/help`",
    ]
    for item in approved_command_catalog():
        lines.append(f"- `/{item['command']}` - {item['description']}")
    lines.extend(
        [
            "",
            "Only slash commands run approved actions. Normal chat remains read-only.",
        ]
    )
    return "\n".join(lines)


def run_assistant_slash_command(question: str) -> tuple[str, dict[str, object] | None]:
    command = question.strip().split()[0].lstrip("/")
    if command in {"", "help", "commands"}:
        return slash_command_help(), None
    allowed = {item["command"] for item in approved_command_catalog()}
    if command not in allowed:
        return f"`/{command}` is not an approved command.\n\n{slash_command_help()}", None
    result = run_approved_command(command)
    if isinstance(result, bytes):
        if not result:
            return f"`/{command}` ran, but there is no data available for that export yet.", None
        filename = COMMAND_OUTPUTS.get(command, f"{command}.zip")
        return f"`/{command}` prepared `{filename}`.", {"label": f"Download {filename}", "file_name": filename, "data": result}
    if isinstance(result, dict):
        return f"`/{command}` result:\n\n```json\n{json.dumps(result, indent=2, ensure_ascii=True)}\n```", None
    return f"`/{command}` result:\n\n{result}", None


def render_assistant_message(message: dict[str, object]) -> None:
    st.write(message["content"])
    download = message.get("download")
    if isinstance(download, dict):
        st.download_button(
            str(download.get("label", "Download file")),
            data=download.get("data", b""),
            file_name=str(download.get("file_name", "download.zip")),
            mime="application/zip",
            use_container_width=True,
            key=f"download_{len(str(download.get('data', b'')))}_{download.get('file_name', '')}_{id(message)}",
        )


def append_assistant_exchange(question: str, answer: str, download: dict[str, object] | None = None) -> None:
    st.session_state["assistant_messages"].append({"role": "user", "content": question})
    assistant_message: dict[str, object] = {"role": "assistant", "content": answer}
    if download:
        assistant_message["download"] = download
    st.session_state["assistant_messages"].append(assistant_message)


def answer_assistant_input(question: str, context: dict[str, object]) -> tuple[str, dict[str, object] | None]:
    if question.strip().startswith("/"):
        return run_assistant_slash_command(question)
    return answer_dataset_question(question, context), None


def submit_assistant_question(question: str, context: dict[str, object]) -> None:
    cleaned = question.strip()
    if not cleaned:
        return
    answer, download = answer_assistant_input(cleaned, context)
    append_assistant_exchange(cleaned, answer, download)


def render_lab_console_controls(context: dict[str, object], quick_questions: list[str]) -> None:
    command_items = approved_command_catalog()
    command_names = [f"/{item['command']}" for item in command_items]
    chips = "".join(f'<span class="command-chip">{html.escape(name)}</span>' for name in ["/help"] + command_names)
    st.markdown(
        f"""
        <div class="assistant-panel">
            <h4>Ask Dr. Bio</h4>
            <p>Use one simple console for questions and approved commands. Click a suggestion, or type your own question below.</p>
            <div>{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="console-label">Common questions</div>', unsafe_allow_html=True)
    for row_start in range(0, len(quick_questions), 2):
        cols = st.columns(2)
        for idx, question in enumerate(quick_questions[row_start : row_start + 2]):
            with cols[idx]:
                if st.button(question, key=f"faq_{row_start}_{idx}", use_container_width=True):
                    submit_assistant_question(question, context)
                    st.rerun()

    st.markdown('<div class="console-label">Approved commands</div>', unsafe_allow_html=True)
    command_buttons = ["/help"] + command_names
    for row_start in range(0, len(command_buttons), 3):
        cols = st.columns(3)
        for idx, command in enumerate(command_buttons[row_start : row_start + 3]):
            with cols[idx]:
                if st.button(command, key=f"cmd_{row_start}_{idx}", use_container_width=True):
                    submit_assistant_question(command, context)
                    st.rerun()

    with st.form("lab_console_form", clear_on_submit=True):
        user_text = st.text_input("Ask Dr. Bio or type /help", placeholder="Example: Which papers were rejected and why?  or  /export_fortrain_zip")
        submitted = st.form_submit_button("Send to Dr. Bio", type="primary", use_container_width=True)
        if submitted:
            submit_assistant_question(user_text, context)
            st.rerun()


def dataset_assistant_tab() -> None:
    render_agent_banner(
        "dr_bio",
        "Dr. Bio / Lab Console",
        "Harness Assistant",
        "Ask about current CSV data, recent rejections, training readiness, or type `/` to see approved lab commands.",
    )
    st.info("Normal chat is read-only. Only explicit slash commands can run approved actions.")

    context = dataset_assistant_context()
    counts = context.get("counts", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Papers", counts.get("papers", 0))
    c2.metric("Materials", counts.get("materials", 0))
    c3.metric("Formulations", counts.get("formulations", 0))
    c4.metric("forTrain rows", counts.get("forTrain_rows", 0))

    quick_questions = [
        "Summarize the current dataset.",
        "What was added in the latest extraction run?",
        "Which formulations are process-only?",
        "Which properties have training data?",
        "Explain the newest paper and its formulations.",
        "Which papers were rejected and why?",
        "How do I export the training data?",
        "/help",
    ]

    if "assistant_messages" not in st.session_state:
        st.session_state["assistant_messages"] = []

    render_lab_console_controls(context, quick_questions)

    for message in st.session_state["assistant_messages"]:
        with st.chat_message(message["role"]):
            render_assistant_message(message)

    if st.button("Clear assistant chat"):
        st.session_state["assistant_messages"] = []
        st.rerun()


def settings_tab() -> None:
    st.header("Settings")
    st.write("OpenAI key lookup: `OPENAI_API_KEY` environment/Streamlit secret, then local `openai_key.txt`.")
    if load_openai_key():
        st.success("OpenAI key detected.")
    else:
        st.warning("No OpenAI key found. Rule-based draft extraction will still work, but OpenAI extraction will be limited.")
    st.write(f"Material dataset: `{FORMULATION_DATASET}`")
    st.write(f"Material library: `{MATERIAL_LIBRARY}`")
    st.write(f"Training CSV folder: `{FOR_TRAIN_DIR}`")

    st.markdown("### API key")
    st.caption("Local development only. Hosted Streamlit deployments should use Streamlit secrets instead.")
    key = st.text_input("OpenAI API key", type="password")
    if st.button("Save API key locally", disabled=not bool(key.strip())):
        OPENAI_KEY_FILE.write_text(key.strip(), encoding="utf-8")
        st.success("Saved OpenAI key locally.")

    st.markdown("### Extraction mode")
    if openai_available():
        st.info("OpenAI-assisted extraction is available.")
    else:
        st.info("OpenAI is not available. Validation can still use local checks, but material/formulation extraction will create no rows rather than guessing.")


def render_sidebar_model_mode() -> None:
    st.subheader("Model Mode")
    mode = st.radio(
        "Run mode",
        ["Online GPT API", "Offline Ollama"],
        index=0,
        horizontal=False,
        key="model_mode",
    )
    if mode == "Online GPT API":
        return

    st.caption("Active when selected. The harness will call this local Ollama server for JSON validation and extraction.")
    st.text_input("Ollama local URL", value="http://localhost:11434", key="ollama_base_url")
    st.text_input("Ollama model", value="qwen2.5:7b", key="ollama_model")
    st.info("Ollama usually needs no API key. Start Ollama locally and pull the model before using future offline mode.")


def apply_streamlit_secrets() -> None:
    if os.environ.get(OPENAI_API_KEY_ENV, "").strip():
        return
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
        if not key and "openai" in st.secrets:
            key = st.secrets["openai"].get("api_key", "")
    except Exception:
        key = ""
    if str(key).strip():
        os.environ[OPENAI_API_KEY_ENV] = str(key).strip()


def sync_drive_storage_once() -> None:
    if st.session_state.get("drive_storage_synced"):
        return
    st.session_state["drive_storage_synced"] = True
    if not drive_configured():
        st.session_state["drive_storage_status"] = {"configured": False, "downloaded": 0}
        return
    try:
        result = sync_from_drive()
        rebuild_source_xlsx_from_csv()
        st.session_state["drive_storage_status"] = {"configured": True, **result}
    except Exception as exc:
        st.session_state["drive_storage_status"] = {"configured": True, "downloaded": 0, "error": str(exc)}


def apply_model_mode_from_sidebar() -> None:
    mode = st.session_state.get("model_mode", "Online GPT API")
    if mode == "Offline Ollama":
        os.environ[LLM_PROVIDER_ENV] = "ollama"
        os.environ[OLLAMA_BASE_URL_ENV] = st.session_state.get("ollama_base_url", "http://localhost:11434").strip() or "http://localhost:11434"
        os.environ[OLLAMA_MODEL_ENV] = st.session_state.get("ollama_model", "qwen2.5:7b").strip() or "qwen2.5:7b"
    else:
        os.environ[LLM_PROVIDER_ENV] = "openai"


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=str(MASCOTS["dr_bio"]), layout="wide")
    apply_streamlit_secrets()
    sync_drive_storage_once()
    inject_css()
    render_header()

    with st.sidebar:
        st.image(str(MASCOTS["dr_bio"]), use_container_width=True)
        st.title("Dr. Bio's Lab")
        st.caption("Extract formulation, material, and property data from papers for ML-ready datasets.")
        st.markdown("---")
        st.metric("Exported papers", len(list_applied_reviews()))
        drive_status = st.session_state.get("drive_storage_status", {})
        current_drive_status = drive_last_status()
        if drive_status.get("configured"):
            if drive_status.get("error") or current_drive_status.get("last_error"):
                st.warning("Google Drive storage needs attention.")
                with st.expander("Drive sync detail"):
                    st.write(drive_status.get("error") or current_drive_status.get("last_error"))
            else:
                st.caption(f"Google Drive CSV storage active. Loaded {drive_status.get('downloaded', 0)} files.")
                if current_drive_status.get("last_upload_path"):
                    st.caption(f"Last Drive upload: {current_drive_status['last_upload_path']}")
        else:
            st.caption("Google Drive CSV storage is not configured.")
        render_sidebar_model_mode()
        apply_model_mode_from_sidebar()
        st.markdown("---")
        render_sidebar_how_it_works()

    tab1, tab2, tab3 = st.tabs(["Sample Intake", "Microscope Bench", "Lab Console"])
    with tab1:
        intake_tab()
    with tab2:
        datasets_tab()
    with tab3:
        dataset_assistant_tab()


if __name__ == "__main__":
    main()
