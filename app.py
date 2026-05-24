from __future__ import annotations

import json
import os
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from extension_loader import list_harness_extensions
from harness_config import APPLIED_LOG, EXTRACTED_SOURCE_CSV, EXTRACTED_SOURCE_XLSX, FORMULATION_COMPONENTS, FOR_TRAIN_DIR, FORMULATION_DATASET, MATERIAL_LIBRARY, MATERIAL_NAME_MAPPING, OPENAI_KEY_FILE, PROPERTY_TARGETS, REJECTION_LOG, UPLOAD_DIR, VALIDATION_LOG
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


APP_TITLE = "Bio Material Harness"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f6f8fb; }
        [data-testid="stSidebar"] { background: #101820; }
        [data-testid="stSidebar"] * { color: #eef5f8; }
        .hero {
            background: #101820;
            color: #ffffff;
            border-radius: 8px;
            padding: 22px 26px;
            margin-bottom: 18px;
            border: 1px solid #314656;
        }
        .hero h1 { margin: 0; font-size: 32px; letter-spacing: 0; }
        .hero p { color: #c8d7df; margin: 8px 0 0; }
        .pill {
            display: inline-block;
            padding: 4px 9px;
            border-radius: 999px;
            background: #d7efe6;
            color: #0e5f46;
            font-size: 12px;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .note {
            border-left: 4px solid #0e7c66;
            background: #ffffff;
            padding: 10px 12px;
            border-radius: 6px;
            color: #334b5a;
            margin: 8px 0 12px;
        }
        .dashboard-title {
            font-size: 22px;
            font-weight: 800;
            color: #14232e;
            margin: 4px 0 2px;
        }
        .dashboard-subtle {
            color: #526579;
            font-size: 14px;
            margin-bottom: 12px;
        }
        .status-strip {
            background: #ffffff;
            border: 1px solid #d9e2ea;
            border-radius: 8px;
            padding: 12px 14px;
            margin: 10px 0 14px;
            color: #243746;
        }
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="pill">Bio Material Data Fractory</div>
            <h1>Bio Material Harness</h1>
            <p>Paper intake, duplicate screening, source extraction, material mapping, and training CSV updates for the BioMaterial platform.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_how_it_works() -> None:
    with st.expander("How it works", expanded=False):
        st.markdown(
            """
            <div class="side-flow">
                <div class="side-step"><span class="side-tag">1. Intake</span><br><strong>Upload paper</strong><br>Read PDF/text. OCR can help when text is weak.</div>
                <div class="side-step"><span class="side-tag">2. Validation Agent</span><br><strong>Check qualification</strong><br>Reject invalid papers, non-biomaterial papers, prompt-injection text, and duplicates.</div>
                <div class="side-step"><span class="side-tag">3. Supplement Check</span><br><strong>Find external data clues</strong><br>Shows Mendeley, Figshare, Zenodo, DOI, or supplementary links when tables may be outside the PDF.</div>
                <div class="side-step"><span class="side-tag">4. Source Extraction</span><br><strong>Save paper metadata</strong><br>Only after real extraction succeeds: title, DOI, authors, summary, confidence, quality, conclusion.</div>
                <div class="side-step"><span class="side-tag">5. Material Agent</span><br><strong>Normalize materials</strong><br>Reuse existing material IDs for same materials. Add new IDs only for new materials.</div>
                <div class="side-step"><span class="side-tag">6. Formulation Agent</span><br><strong>Extract formulation/process rows</strong><br>Composition, wt%, roles, processing temperature, speed, feeding rate, time, and related evidence.</div>
                <div class="side-step"><span class="side-tag">7. Training Builder</span><br><strong>Create forTrain rows</strong><br>Only measured property rows go into training CSVs. Process-only rows stay useful but are marked not training-ready.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Rule: no evidence-backed material/formulation data means no CSV update.")


def render_loaded_extensions() -> None:
    extensions = list_harness_extensions()
    app_dir = Path(__file__).resolve().parent
    with st.expander("Loaded extensions", expanded=False):
        if not extensions:
            st.caption("No local skills found.")
            return
        for extension in extensions:
            st.markdown(f"**{extension.name}**")
            if extension.summary:
                st.caption(extension.summary)
            try:
                display_path = extension.path.resolve().relative_to(app_dir)
            except ValueError:
                display_path = extension.path
            st.caption(f"`{display_path}`")


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
    st.header("Paper Validation Agent")
    st.markdown(
        '<div class="note">Upload a research paper first. The validation agent checks paper quality, biomaterial/formulation relevance, and duplicates before extraction runs.</div>',
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
    st.success("Qualified unique biomaterial formulation paper.")
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
                    st.success("Extraction pipeline complete. Source, datasets, and forTrain are updated.")
                else:
                    st.success("Formulation/process data extracted. Source and datasets are updated.")
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


def zip_existing_files(files: list[tuple[str, Path]]) -> bytes:
    existing_files = [(arcname, path) for arcname, path in files if path.exists() and path.is_file()]
    if not existing_files:
        return b""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for arcname, path in existing_files:
            archive.writestr(arcname, path.read_bytes())
    buffer.seek(0)
    return buffer.getvalue()


def train_export_files() -> list[tuple[str, Path]]:
    return [(f"forTrain/{path.name}", path) for path in sorted(FOR_TRAIN_DIR.glob("*.csv"))]


def dataset_export_files() -> list[tuple[str, Path]]:
    return [
        ("data/extraction/biomaterial_source_papers.csv", EXTRACTED_SOURCE_CSV),
        ("data/extraction/biomaterial_source_papers.xlsx", EXTRACTED_SOURCE_XLSX),
        ("data/datasets/material_library.csv", MATERIAL_LIBRARY),
        ("data/datasets/material_name_mapping.csv", MATERIAL_NAME_MAPPING),
        ("data/datasets/formulation_dataset.csv", FORMULATION_DATASET),
        ("data/datasets/formulation_components.csv", FORMULATION_COMPONENTS),
    ]


def audit_export_files() -> list[tuple[str, Path]]:
    files = dataset_export_files()
    files.extend(
        [
            ("data/validation/validation_log.csv", VALIDATION_LOG),
            ("data/validation/rejected_papers.csv", REJECTION_LOG),
            ("data/logs/applied_reviews.csv", APPLIED_LOG),
        ]
    )
    files.extend(train_export_files())
    return files


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
    st.markdown('<div class="dashboard-title">Data Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtle">Explore the local paper, material, formulation, component, and training CSVs from one place.</div>', unsafe_allow_html=True)
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

    overview, explorer, relationships, training, files = st.tabs(["Dashboard", "Explore Data", "Relationships", "Training", "Files"])

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
        st.subheader("Export")
        export_cols = st.columns(3)
        train_zip = zip_existing_files(train_export_files())
        dataset_zip = zip_existing_files(dataset_export_files())
        audit_zip = zip_existing_files(audit_export_files())
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
        "Exports are available in `Data Overview` -> `Files`.\n\n"
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


def dataset_assistant_tab() -> None:
    st.markdown('<div class="dashboard-title">Dataset Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtle">Read-only assistant for the current CSV data. It can explain papers, materials, formulations, relationships, and training readiness.</div>', unsafe_allow_html=True)
    st.info("Read-only mode: this assistant cannot edit, approve, delete, extract, or update CSV files.")

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
    ]
    selected = st.selectbox("Quick question", [""] + quick_questions)
    typed = st.chat_input("Ask about papers, materials, formulations, or training CSVs")
    question = typed or selected

    if "assistant_messages" not in st.session_state:
        st.session_state["assistant_messages"] = []

    for message in st.session_state["assistant_messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if question:
        st.session_state["assistant_messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Reading CSV snapshot..."):
                answer = answer_dataset_question(question, context)
            st.write(answer)
        st.session_state["assistant_messages"].append({"role": "assistant", "content": answer})

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


def apply_model_mode_from_sidebar() -> None:
    mode = st.session_state.get("model_mode", "Online GPT API")
    if mode == "Offline Ollama":
        os.environ[LLM_PROVIDER_ENV] = "ollama"
        os.environ[OLLAMA_BASE_URL_ENV] = st.session_state.get("ollama_base_url", "http://localhost:11434").strip() or "http://localhost:11434"
        os.environ[OLLAMA_MODEL_ENV] = st.session_state.get("ollama_model", "qwen2.5:7b").strip() or "qwen2.5:7b"
    else:
        os.environ[LLM_PROVIDER_ENV] = "openai"


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="MS", layout="wide")
    apply_streamlit_secrets()
    inject_css()
    render_header()

    with st.sidebar:
        st.title("BIO Material Harness")
        st.caption("Validate papers, extract formulations, and build training CSVs.")
        st.markdown("---")
        st.metric("Extracted papers", len(list_applied_reviews()))
        render_sidebar_model_mode()
        apply_model_mode_from_sidebar()
        st.markdown("---")
        render_loaded_extensions()
        st.markdown("---")
        render_sidebar_how_it_works()

    tab1, tab2, tab3 = st.tabs(["Paper Intake", "Data Overview", "Dataset Assistant"])
    with tab1:
        intake_tab()
    with tab2:
        datasets_tab()
    with tab3:
        dataset_assistant_tab()


if __name__ == "__main__":
    main()
