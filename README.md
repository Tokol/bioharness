# Bio Material Harness

Bio Material Harness is a local agentic data pipeline for turning biomaterial research papers into structured CSV datasets for future machine learning.

The project was built as a domain-specific agent harness: instead of transforming markdown into slides, it transforms research-paper evidence into biomaterial source records, material libraries, formulation rows, component mappings, and `forTrain` model-training CSVs.

## Architecture at a Glance

The harness is layered so the model is never the only authority. The UI, validation rules, extraction agents, CSV writers, logs, and read-only assistant each have separate responsibilities.

![Layered Bio Material Agent Harness](images/figure_1_layered_agent_harness.png)

The operational flow starts with paper upload and ends only when evidence-backed data is safely written to local CSVs or rejected with a logged reason.

![Bio Material Agent Workflow](images/figure_2_agent_workflow.png)

## What This Is

This app helps collect biomaterial formulation data from papers without blindly trusting the model.

It can:

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

## Why This Exists

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

## Agent Flow

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

## Online and Offline Modes

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

## Installation

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

For local development, create a local file:

```text
openai_key.txt
```

Put only the key inside that file.

Do not commit this file.

For Streamlit Cloud or other hosted deployment, do not use `openai_key.txt`. Add the key through Streamlit secrets instead:

```toml
OPENAI_API_KEY = "sk-..."
```

The app checks keys in this order:

1. `OPENAI_API_KEY` from environment variables or Streamlit secrets
2. local `openai_key.txt`

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

## GitHub and Deployment Notes

Do not commit secrets or generated paper-derived data unless you intentionally want them public.

Recommended `.gitignore` entries:

```gitignore
openai_key.txt
.env
__pycache__/
.venv/
*.zip

data/uploads/*
data/review/*
data/applied_reviews/*
data/extraction/*
data/validation/*
data/logs/*
data/datasets/*.csv
data/datasets/*.xlsx
forTrain/*.csv
```

Keep empty folders with `.gitkeep` files if needed.

For a public demo, prefer synthetic sample data under a separate `sample_data/` folder instead of real extracted paper data.

## How This Fits the Assignment Theme

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

Partially implemented. During development, compile checks and smoke checks are run manually. The app also performs strict validation and dataset checks before writing CSVs.

A future improvement would add a full automatic post-extraction audit loop:

- run consistency checks after extraction
- reject or quarantine failed rows
- optionally retry extraction once
- log pass/fail audit results

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

This harness reads files, processes untrusted paper text, calls LLMs, and writes local CSV datasets. Those are real risks. This section describes the attack surface and what the project does about it.

### File Operations Escaping the Working Directory

Risk:

- A file-processing tool could read or write outside the project.
- A malicious upload name could try path traversal.
- Exports could include unintended files.

Mitigation:

- The app writes to known project directories such as `data/` and `forTrain/`.
- Uploaded papers are saved temporarily under `data/uploads/` and deleted after text extraction.
- Export ZIPs are built from explicit allowlists of known CSV/XLSX paths.
- The assistant does not receive file-write tools.

Accepted risk:

- This is a local Streamlit app, not a hardened multi-user server. If deployed publicly, upload path handling and file-size limits should be hardened further.

### Shell Commands Constructed From Tool Output

Risk:

- If an agent can build shell commands from untrusted model/tool output, prompt injection can become code execution.

Mitigation:

- The Streamlit app does not expose shell execution to the model or Dataset Assistant.
- The Dataset Assistant can only read CSV-derived context.
- Exports are controlled UI buttons, not assistant-triggered shell commands.

Accepted risk:

- Developers may run shell commands manually during development. That is outside the app's runtime tool surface.

### Prompt Injection From Files

Risk:

- A PDF may contain text such as “ignore previous instructions” or fake extraction instructions.
- A paper may include keywords like biomaterial/formulation without real evidence.
- Supplementary files may contain misleading labels.

Mitigation:

- Validation and extraction prompts explicitly treat paper text as untrusted evidence.
- Prompt-injection-like terms are detected and can cap/reject validation.
- Extraction requires evidence from the study, methods, results, figures, tables, or user-uploaded supplementary tables.
- References, background text, metadata tags, future work, and author instructions are not accepted as extraction evidence.
- Invalid and duplicate decisions are logged to `data/validation/rejected_papers.csv`.

Accepted risk:

- LLMs can still make mistakes. The harness reduces damage by requiring evidence fields and by not saving when required rows are missing.

### Secrets in Environment Variables or Files

Risk:

- API keys could leak into model context, logs, GitHub, or exports.

Mitigation:

- Online mode reads `OPENAI_API_KEY` from environment variables or Streamlit secrets first, then falls back to local `openai_key.txt`.
- The Dataset Assistant context is built from CSV data only, not environment variables or secret files.
- Export bundles do not include `openai_key.txt`.
- README recommends ignoring `openai_key.txt`, `.env`, generated data, and ZIP exports.

Accepted risk:

- The local key file may exist during development. Users must keep it out of Git and deployment bundles.

### Untrusted MCP Servers and Skills

Risk:

- A harness that loads arbitrary MCP servers or skills can expand its tool surface unexpectedly.

Mitigation:

- This project does not load MCP servers.
- This project does not load dynamic skills from a folder.
- The model has no direct access to external tools beyond the selected LLM provider call.

Accepted risk:

- MCP/skill plugin loading is out of scope for this app. If added later, it should require explicit allowlists and user confirmation.

### Supplementary ZIP/XLSX/CSV Uploads

Risk:

- ZIP files can contain huge files or many nested entries.
- Tables can contain irrelevant or malicious text.

Mitigation:

- Only CSV/XLSX/XLS files are parsed from ZIP uploads.
- ZIP members larger than 25 MB are skipped.
- Supplementary text is capped before being sent to the model.
- Supplementary evidence is labeled explicitly in the extraction prompt.
- Data is only saved after evidence-backed extraction succeeds.

Accepted risk:

- More robust scanning would be needed for hostile public uploads.

### Read-Only Dataset Assistant

Risk:

- A user could ask the assistant to change data or fabricate rows.

Mitigation:

- The assistant only receives a CSV-derived snapshot.
- It has no write, delete, export-trigger, or extraction functions.
- Rejection/export questions use deterministic code paths instead of free model interpretation.
- The system prompt tells the assistant to refuse edits and answer only from supplied data.

Accepted risk:

- The assistant may answer imperfectly for general questions. It cannot mutate project data.

## Current Limitations

- OCR requires system Tesseract to be installed.
- Offline Ollama quality depends on the local model.
- Evaluation loop is not fully automated yet.
- Row-level timestamps are not stored on every material/formulation row; latest run history is available from logs.
- Public deployment should add stricter upload limits, auth, and stronger file isolation.

## License and Data Responsibility

This repository should contain code, not private API keys or copyrighted/generated paper-derived datasets unless you have permission to publish them.

Generated CSVs are intended for local research workflow and should be reviewed before sharing publicly.
