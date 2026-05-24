from __future__ import annotations

import argparse
import json
import zipfile
from io import BytesIO
from pathlib import Path

from harness_config import (
    APPLIED_LOG,
    EXTRACTED_SOURCE_CSV,
    EXTRACTED_SOURCE_XLSX,
    FORMULATION_COMPONENTS,
    FORMULATION_DATASET,
    FOR_TRAIN_DIR,
    MATERIAL_LIBRARY,
    MATERIAL_NAME_MAPPING,
    PROPERTY_TARGETS,
    REJECTION_LOG,
    VALIDATION_LOG,
)
from utils import read_table


COMMAND_OUTPUTS = {
    "export_fortrain_zip": "biomaterial_forTrain.zip",
    "export_dataset_zip": "biomaterial_datasets.zip",
    "export_audit_bundle_zip": "biomaterial_audit_bundle.zip",
}


def zip_existing_files(files: list[tuple[str, Path]]) -> bytes:
    existing_files = [(arcname, path) for arcname, path in files if exportable_file_has_data(path)]
    if not existing_files:
        return b""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for arcname, path in existing_files:
            archive.writestr(arcname, path.read_bytes())
    buffer.seek(0)
    return buffer.getvalue()


def exportable_file_has_data(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if path.suffix.lower() in {".csv", ".xlsx", ".xls"}:
        try:
            return not read_table(path).empty
        except Exception:
            return False
    return path.stat().st_size > 0


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


def export_fortrain_zip() -> bytes:
    return zip_existing_files(train_export_files())


def export_dataset_zip() -> bytes:
    return zip_existing_files(dataset_export_files())


def export_audit_bundle_zip() -> bytes:
    return zip_existing_files(audit_export_files())


def validation_smoke_check() -> dict[str, object]:
    checks = []
    for label, path in [
        ("source_csv", EXTRACTED_SOURCE_CSV),
        ("materials", MATERIAL_LIBRARY),
        ("formulations", FORMULATION_DATASET),
        ("components", FORMULATION_COMPONENTS),
        ("rejected_papers", REJECTION_LOG),
    ]:
        checks.append({"check": label, "path": str(path), "exists": path.exists(), "rows": len(read_table(path)) if path.exists() else 0})
    checks.append({"check": "configured_property_targets", "path": "harness_config.PROPERTY_TARGETS", "exists": True, "rows": len(PROPERTY_TARGETS)})
    return {"status": "ok", "checks": checks}


APPROVED_COMMANDS: dict[str, dict[str, object]] = {
    "export_fortrain_zip": {
        "description": "Create a ZIP snapshot of ML training CSVs from forTrain/.",
        "writes_data": False,
        "runner": export_fortrain_zip,
    },
    "export_dataset_zip": {
        "description": "Create a ZIP snapshot of source, material, formulation, component, and mapping files.",
        "writes_data": False,
        "runner": export_dataset_zip,
    },
    "export_audit_bundle_zip": {
        "description": "Create a ZIP snapshot of datasets plus validation, rejection, applied-run, and training logs.",
        "writes_data": False,
        "runner": export_audit_bundle_zip,
    },
    "validation_smoke_check": {
        "description": "Read known dataset paths and report whether expected files exist.",
        "writes_data": False,
        "runner": validation_smoke_check,
    },
}


def approved_command_catalog() -> list[dict[str, object]]:
    return [
        {"command": name, "description": str(meta["description"]), "writes_data": bool(meta["writes_data"])}
        for name, meta in APPROVED_COMMANDS.items()
    ]


def run_approved_command(name: str) -> object:
    if name not in APPROVED_COMMANDS:
        raise ValueError(f"Command is not approved: {name}")
    runner = APPROVED_COMMANDS[name]["runner"]
    if not callable(runner):
        raise ValueError(f"Approved command is not callable: {name}")
    return runner()


def explain_commands() -> str:
    lines = [
        "Approved harness commands:",
        "",
    ]
    for item in approved_command_catalog():
        lines.append(f"- {item['command']}: {item['description']} Writes data: {item['writes_data']}.")
    lines.extend(
        [
            "",
            "Limits:",
            "- Natural language is only mapped to this allowlist.",
            "- No shell commands are executed.",
            "- Export commands only read allowlisted project CSV/XLSX files.",
            "- If the matching files are empty or missing, the command explains that instead of failing.",
        ]
    )
    return "\n".join(lines)


def command_from_prompt(prompt: str) -> str | None:
    text = prompt.lower()
    if any(term in text for term in ["what command", "which command", "help", "list command", "what can you do", "limitation", "limit"]):
        return "help"
    if any(term in text for term in ["smoke", "health", "check", "validate files", "validation check"]):
        return "validation_smoke_check"
    if any(term in text for term in ["audit", "logs", "rejection", "rejected", "bundle"]):
        return "export_audit_bundle_zip"
    if any(term in text for term in ["fortrain", "for train", "training data", "ml data", "machine learning"]):
        return "export_fortrain_zip"
    if any(term in text for term in ["dataset", "datasets", "materials", "formulation", "component", "mapping"]):
        return "export_dataset_zip"
    if any(term in text for term in ["export", "download", "zip"]):
        return "help"
    return None


def command_is_empty(name: str, result: object) -> bool:
    if isinstance(result, bytes):
        return len(result) == 0
    if name == "validation_smoke_check" and isinstance(result, dict):
        return False
    return result in (None, "", [], {})


def run_prompt(prompt: str, output_dir: Path | None = None) -> str:
    command = command_from_prompt(prompt)
    if command in {None, "help"}:
        return explain_commands()

    result = run_approved_command(command)
    if command_is_empty(command, result):
        return f"`{command}` matched your request, but there is no data available for that export yet."

    if isinstance(result, bytes):
        output_dir = output_dir or Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = COMMAND_OUTPUTS.get(command, f"{command}.zip")
        output_path = output_dir / filename
        output_path.write_bytes(result)
        return f"Ran `{command}`. Wrote `{output_path}`."

    if isinstance(result, dict):
        return f"Ran `{command}`.\n{json.dumps(result, indent=2, ensure_ascii=True)}"

    return f"Ran `{command}`.\n{result}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run approved Bio Material Harness commands from a natural-language prompt.")
    parser.add_argument("prompt", nargs="*", help="Natural-language request, for example: export forTrain data")
    parser.add_argument("--out", default=".", help="Output folder for ZIP exports. Default: current folder.")
    args = parser.parse_args()

    prompt = " ".join(args.prompt).strip()
    if not prompt:
        prompt = "help"
    print(run_prompt(prompt, Path(args.out)))


if __name__ == "__main__":
    main()
