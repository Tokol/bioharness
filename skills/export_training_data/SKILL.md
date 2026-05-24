# Export Training Data Skill

Purpose: define the safe export behavior for downstream machine-learning work.

The harness implementation lives in `harness_commands.py`; this skill file is the extension specification for the export command.

## Approved Command

```text
export_fortrain_zip
```

## Inputs

- existing CSV files under `forTrain/`

## Rules

- Read only existing training CSVs.
- Do not edit, delete, approve, or regenerate dataset rows.
- Do not include API keys, uploads, PDFs, or private local files.
- Export only files selected by the allowlist in `harness_commands.py`.

## Output

- `biomaterial_forTrain.zip`, created from current training CSV snapshots.
