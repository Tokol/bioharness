from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from harness_config import (
    APPLIED_LOG,
    BASE_DIR,
    EXTRACTED_SOURCE_CSV,
    FORMULATION_COMPONENTS,
    FORMULATION_DATASET,
    MATERIAL_LIBRARY,
    MATERIAL_NAME_MAPPING,
    PROPERTY_TARGETS,
    REJECTION_LOG,
    VALIDATION_LOG,
)


DRIVE_SYNC_DISABLED_ENV = "BIOMATERIAL_DISABLE_DRIVE_SYNC"
LAST_STATUS: dict[str, Any] = {
    "configured": False,
    "last_upload_path": "",
    "last_upload_at": "",
    "last_error": "",
}


class DriveSyncError(RuntimeError):
    pass


def last_status() -> dict[str, Any]:
    status = dict(LAST_STATUS)
    status["configured"] = drive_configured()
    status["has_folder_id"] = bool(get_drive_folder_id())
    status["has_service_account"] = bool(get_service_account_info())
    return status


def missing_config_labels() -> list[str]:
    missing = []
    if not get_drive_folder_id():
        missing.append("GOOGLE_DRIVE_FOLDER_ID")
    if not get_service_account_info():
        missing.append("[gcp_service_account]")
    return missing


def _remember_upload(path: Path) -> None:
    LAST_STATUS.update(
        {
            "configured": True,
            "last_upload_path": relative_drive_path(path),
            "last_upload_at": datetime.now().isoformat(timespec="seconds"),
            "last_error": "",
        }
    )


def _remember_error(exc: Exception) -> None:
    LAST_STATUS["last_error"] = str(exc)


def managed_csv_paths() -> list[Path]:
    paths = [
        EXTRACTED_SOURCE_CSV,
        FORMULATION_DATASET,
        FORMULATION_COMPONENTS,
        MATERIAL_LIBRARY,
        MATERIAL_NAME_MAPPING,
        VALIDATION_LOG,
        REJECTION_LOG,
        APPLIED_LOG,
    ]
    paths.extend(FORMULATION_DATASET.parent / filename for filename in PROPERTY_TARGETS.values())
    paths.extend((BASE_DIR / "forTrain") / filename for filename in PROPERTY_TARGETS.values())
    return paths


def relative_drive_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(BASE_DIR.resolve()).as_posix()
    except ValueError:
        return path.name


def _streamlit_secret(name: str, default: Any = "") -> Any:
    try:
        import streamlit as st

        return st.secrets.get(name, default)
    except Exception:
        return default


def drive_configured() -> bool:
    return bool(get_drive_folder_id() and get_service_account_info())


def get_drive_folder_id() -> str:
    return str(os.environ.get("GOOGLE_DRIVE_FOLDER_ID") or _streamlit_secret("GOOGLE_DRIVE_FOLDER_ID", "")).strip()


def get_service_account_info() -> dict[str, Any]:
    try:
        info = _streamlit_secret("gcp_service_account", {})
        if info:
            return dict(info)
    except Exception:
        pass
    return {}


def _drive_service():
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build

    info = get_service_account_info()
    private_key = info.get("private_key")
    if isinstance(private_key, str):
        info["private_key"] = private_key.replace("\\n", "\n")
    credentials = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/drive"])
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def _find_child(service: Any, parent_id: str, name: str, mime_type: str | None = None) -> str:
    escaped = name.replace("'", "\\'")
    query = f"name = '{escaped}' and '{parent_id}' in parents and trashed = false"
    if mime_type:
        query += f" and mimeType = '{mime_type}'"
    result = service.files().list(q=query, fields="files(id, name)", pageSize=1, supportsAllDrives=True).execute()
    files = result.get("files", [])
    return files[0]["id"] if files else ""


def _ensure_folder(service: Any, parent_id: str, name: str) -> str:
    folder_mime = "application/vnd.google-apps.folder"
    existing = _find_child(service, parent_id, name, folder_mime)
    if existing:
        return existing
    metadata = {"name": name, "mimeType": folder_mime, "parents": [parent_id]}
    created = service.files().create(body=metadata, fields="id", supportsAllDrives=True).execute()
    return created["id"]


def _parent_for_relative_path(service: Any, root_folder_id: str, rel_path: str) -> str:
    parent_id = root_folder_id
    parts = rel_path.split("/")[:-1]
    for part in parts:
        parent_id = _ensure_folder(service, parent_id, part)
    return parent_id


def upload_csv(path: Path) -> bool:
    try:
        if os.environ.get(DRIVE_SYNC_DISABLED_ENV) == "1" or path.suffix.lower() != ".csv" or not path.exists():
            return False
        if not drive_configured():
            return False
        from googleapiclient.http import MediaFileUpload

        service = _drive_service()
        root_folder_id = get_drive_folder_id()
        rel_path = relative_drive_path(path)
        parent_id = _parent_for_relative_path(service, root_folder_id, rel_path)
        file_name = Path(rel_path).name
        media = MediaFileUpload(str(path), mimetype="text/csv", resumable=False)
        existing_id = _find_child(service, parent_id, file_name)
        if existing_id:
            service.files().update(fileId=existing_id, media_body=media, fields="id", supportsAllDrives=True).execute()
        else:
            service.files().create(body={"name": file_name, "parents": [parent_id]}, media_body=media, fields="id", supportsAllDrives=True).execute()
        _remember_upload(path)
        return True
    except Exception as exc:
        _remember_error(exc)
        raise DriveSyncError(f"Google Drive upload failed for {relative_drive_path(path)}: {exc}") from exc


def download_csv(path: Path) -> bool:
    if not drive_configured():
        return False
    from googleapiclient.http import MediaIoBaseDownload
    from io import BytesIO

    service = _drive_service()
    root_folder_id = get_drive_folder_id()
    rel_path = relative_drive_path(path)
    parent_id = root_folder_id
    for part in rel_path.split("/")[:-1]:
        parent_id = _find_child(service, parent_id, part, "application/vnd.google-apps.folder")
        if not parent_id:
            return False
    file_id = _find_child(service, parent_id, Path(rel_path).name)
    if not file_id:
        return False

    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buffer = BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())
    return True


def sync_from_drive() -> dict[str, int]:
    if not drive_configured():
        return {"downloaded": 0, "available": 0}
    downloaded = 0
    for path in managed_csv_paths():
        try:
            downloaded += int(download_csv(path))
        except Exception:
            continue
    return {"downloaded": downloaded, "available": len(managed_csv_paths())}


def rebuild_source_xlsx_from_csv() -> bool:
    from harness_config import EXTRACTED_SOURCE_XLSX

    if not EXTRACTED_SOURCE_CSV.exists():
        return False
    try:
        pd.read_csv(EXTRACTED_SOURCE_CSV).to_excel(EXTRACTED_SOURCE_XLSX, index=False)
        return True
    except Exception:
        return False
