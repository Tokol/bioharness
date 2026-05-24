# Bio Material Harness

Bio Material Harness is a local agentic data pipeline for turning biomaterial research papers into structured CSV datasets for future machine learning.

The project was built as a domain-specific agent harness: instead of transforming markdown into slides, it transforms research-paper evidence into biomaterial source records, material libraries, formulation rows, component mappings, and `forTrain` model-training CSVs.

Live app: https://bioharness.streamlit.app/

## Architecture at a Glance

The harness is layered so the model is never the only authority. The UI, validation rules, extraction agents, CSV writers, logs, and read-only assistant each have separate responsibilities.

![Layered Bio Material Agent Harness](images/figure_1_layered_agent_harness.png)

The operational flow starts with paper upload and ends only when evidence-backed data is safely written to local CSVs or rejected with a logged reason.

![Bio Material Agent Workflow](images/figure_2_agent_workflow.png)

## What It Does

Bio Material Harness helps collect biomaterial formulation data without blindly trusting the model. It can:

- Upload a PDF/TXT/MD research paper.
- Validate whether it is a real biomaterial/formulation paper.
- Reject irrelevant papers and log why.
- Detect duplicates against already extracted source papers.
- Detect supplementary data links in valid papers.
- Accept manual supplementary CSV/XLSX/ZIP uploads.
- Extract materials, formulation/process rows, components, and measured properties.
- Separate process-only formulations from training-ready property rows.
- Normalize repeated materials so the same material keeps the same `material_id`.
- Generate local training CSVs under `forTrain/`.
- Provide a read-only Dataset Assistant for questions about current CSV data.
- Export dataset, training, or audit bundles as ZIP files.

## Why

Scientific papers are messy. Important formulation data can be in tables, paragraphs, images, or supplementary files. A plain chatbot answer is not enough because the output becomes training data.

The harness adds control around the model:

- validation before extraction
- duplicate checks before saving
- evidence requirements before CSV updates
- rejection logs for failed papers
- read-only assistant boundaries
- local export bundles for reproducible ML datasets

The core rule is:

> No evidence-backed material/formulation data means no dataset update.

## Flow

1. **Paper Intake**
   - Reads PDF, TXT, or pasted paper text.
   - Uses OCR fallback when configured and needed.
   - Deletes uploaded PDFs after text extraction.

2. **Paper Validation Agent**
   - Checks whether the document is paper-like.
   - Checks biomaterial, formulation, experimental, and property signals.
   - Detects prompt-injection-like text.
   - Rejects invalid, irrelevant, or duplicate papers.

3. **Supplementary Data Check**
   - For qualified papers only, finds likely supplementary/data repository links.
   - Lets the user manually upload CSV/XLSX/ZIP supplementary files.

4. **Source Extraction**
   - Saves paper metadata only after extraction succeeds.
   - Records title, DOI, authors, summary, conclusion, confidence, and quality.

5. **Material/Formulation Extraction Agent**
   - Extracts materials, roles, wt%, formulation codes, process variables, and measured properties.
   - Reuses existing material IDs when the material is already known.
   - Saves process-only formulations even when measured properties are missing.

6. **Training Builder**
   - Builds `forTrain` rows only from measured property values.
   - Process-only rows stay useful in the formulation dataset, but they are not used as ML targets.

7. **Dataset Assistant**
   - Read-only assistant for current CSVs.
   - Answers questions about papers, materials, formulations, rejected papers, and training data.
   - Cannot modify files or run extraction.

## Model Modes

The sidebar has a model mode selector.

### Online GPT API

Default mode. Uses the OpenAI API with the configured key.

### Offline Ollama

Offline mode calls a local Ollama server:

```text
http://localhost:11434
```

Example model:

```text
qwen2.5:7b
```

Ollama normally does not need an API key. The model must be installed locally, for example:

```bash
ollama pull qwen2.5:7b
```

The same validation and CSV safety rules apply in both modes. If the local model fails or returns bad JSON, the harness returns no model data and avoids unsafe CSV updates.

## Data Outputs

Generated local outputs include:

```text
data/extraction/biomaterial_source_papers.csv
data/extraction/biomaterial_source_papers.xlsx
data/datasets/material_library.csv
data/datasets/material_name_mapping.csv
data/datasets/formulation_dataset.csv
data/datasets/formulation_components.csv
data/validation/validation_log.csv
data/validation/rejected_papers.csv
data/logs/applied_reviews.csv
forTrain/*.csv
```

The current property targets are configured in `harness_config.py`.

## Process-Only vs Training-Ready

A paper can be useful even without measured property values.

**Process-only** means:

- material/formulation/process evidence exists
- measured property values are missing or not linked
- rows are saved to formulation/material datasets
- no ML target row is added to `forTrain`

**Training-ready** means:

- material/formulation/process evidence exists
- measured property values exist
- property target rows can be generated in `forTrain`

## Exports

`Data Overview -> Files` provides three read-only exports:

- **Download forTrain ZIP**
  - only ML training CSVs from `forTrain/`

- **Download dataset ZIP**
  - source paper, material, formulation, component, and mapping files

- **Download audit bundle ZIP**
  - datasets, validation logs, rejection logs, applied run logs, and `forTrain` CSVs

The Dataset Assistant can explain where the export buttons are, but it does not trigger downloads itself.

## Local Installation

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd BioMaterialHarness
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
. .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Add an OpenAI key for online mode

Create a local file:

```text
openai_key.txt
```

Put only the key inside that file.

Do not commit this file.

### 5. Optional OCR setup

The Python OCR packages are in `requirements.txt`, but Tesseract itself must be installed on the system.

On macOS with Homebrew:

```bash
brew install tesseract
```

If Tesseract is not installed, normal PDF text extraction still works, but OCR fallback will show a clear error when needed.

### 6. Run the app

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## GitHub Notes

Do not commit secrets or generated paper-derived data. This repo ignores `openai_key.txt`, virtual environments, uploads, generated CSV/XLSX files, logs, validation data, applied review archives, ZIP exports, and `forTrain/*.csv`. Empty folders are kept with `.gitkeep`.

## Assignment Theme

The assignment example describes a harness where an agent talks to an LLM, uses tools, reads/writes files, and transforms one artifact into another.

This project maps that idea to biomaterial data collection:

```text
Research paper PDF + supplementary tables
        -> validation
        -> source metadata
        -> material library
        -> formulation/component datasets
        -> training CSVs for ML
```

It uses a staged harness rather than an unconstrained chatbot. The agent roles are separated:

- Paper Validation Agent
- Source Extraction
- Material/Formulation Extraction
- Training Builder
- Dataset Assistant

The harness decides when data can be saved. The model proposes structured JSON, but CSV mutation is guarded by code.

## Inspiration Features

### Offline

Implemented through Ollama mode. The harness can call a local model at `localhost:11434`, so extraction can run without cloud API calls when a local model is available.

### Safety Interlock

Implemented as domain gates:

- invalid paper rejection
- duplicate detection
- prompt-injection detection
- evidence-required extraction
- source record saved only after successful extraction
- training rows only from measured property values
- rejection log for invalid and duplicate papers
- read-only Dataset Assistant

Whitelist-style exceptions are intentionally narrow. For example, process-only formulation rows can be accepted without measured properties, but only when source evidence exists.

### Evaluation Loop

Partially implemented. During development, compile checks and smoke checks are run manually. The app also performs strict validation and dataset checks before writing CSVs. A future improvement would add a full automatic post-extraction audit/retry loop.

### Context Management

Implemented through the Dataset Assistant. Instead of feeding all history to a model, the app builds a compact CSV-derived snapshot:

- counts
- latest extraction run
- recent papers
- materials
- formulations
- relationships
- process-only rows
- training inventory
- recent rejections

The assistant answers from that summary and current CSV records.

### Multi-Agent

Implemented as a staged agentic pipeline. It is not a free-form planner/executor/critic swarm. The harness assigns responsibilities:

- validation decides whether a paper can proceed
- extraction proposes rows
- normalization maps materials
- training builder generates ML-ready CSVs
- read-only assistant explains existing data

This keeps the system easier to audit.

## Security Discussion

This harness reads files, processes untrusted paper text, calls LLMs, and writes local CSV datasets. The main risks and mitigations are:

- **File access:** uploads and generated files stay under known project folders; PDFs are deleted after text extraction; ZIP exports use explicit allowlists.
- **Shell execution:** the app does not expose shell execution to the model or Dataset Assistant.
- **Prompt injection:** paper text is treated as untrusted evidence; injection-like text is detected; extraction ignores references, background, metadata tags, and author instructions.
- **Secrets:** `OPENAI_API_KEY` or `openai_key.txt` is never included in assistant context or export bundles, and key files are ignored by git.
- **Untrusted MCP/skills:** this app does not load MCP servers or dynamic skills.
- **Supplementary uploads:** ZIP parsing is limited to CSV/XLS/XLSX, large members are skipped, supplementary text is capped, and rows are saved only after evidence-backed extraction.
- **Read-only assistant:** the assistant only receives CSV-derived context and has no write, delete, export-trigger, or extraction functions.

Accepted risks: this is a local/research Streamlit app, not a hardened multi-user service. Public use should add stricter upload limits, authentication, and stronger file isolation. LLMs can still make mistakes, so the harness reduces damage by requiring evidence and refusing CSV updates when checks fail.

## Current Limitations

- OCR requires system Tesseract to be installed.
- Offline Ollama quality depends on the local model.
- Evaluation loop is not fully automated yet.
- Row-level timestamps are not stored on every material/formulation row; latest run history is available from logs.
- Public use should add stricter upload limits, auth, and stronger file isolation.

## License and Data Responsibility

This repository should contain code, not private API keys or copyrighted/generated paper-derived datasets unless you have permission to publish them.

Generated CSVs are intended for local research workflow and should be reviewed before sharing publicly.
