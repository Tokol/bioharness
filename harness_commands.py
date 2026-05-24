from __future__ import annotations

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
