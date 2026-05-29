from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

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
SERVICE_ACCOUNT_JSON_SECRET_NAMES = [
    "GCP_SERVICE_ACCOUNT_JSON",
    "GOOGLE_SERVICE_ACCOUNT_JSON",
    "gcp_service_account_json",
]
LAST_STATUS: dict[str, Any] = {
    "configured": False,
    "backend": "",
    "last_upload_path": "",
    "last_upload_at": "",
    "last_error": "",
}


class DriveSyncError(RuntimeError):
    pass


class RemoteStorageError(RuntimeError):
    pass


def last_status() -> dict[str, Any]:
    status = dict(LAST_STATUS)
    status["configured"] = remote_storage_configured()
    status["backend"] = active_backend()
    status["has_supabase_url"] = bool(get_supabase_url())
    status["has_supabase_key"] = bool(get_supabase_key())
    status["has_supabase_bucket"] = bool(get_supabase_bucket())
    status["has_folder_id"] = bool(get_drive_folder_id())
    status["has_service_account"] = bool(get_service_account_info())
    return status


def missing_config_labels() -> list[str]:
    if supabase_configured() or drive_configured():
        return []
    if get_supabase_url() or get_supabase_key() or get_supabase_bucket():
        missing = []
        if not get_supabase_url():
            missing.append("SUPABASE_URL")
        if not get_supabase_key():
            missing.append("SUPABASE_SERVICE_ROLE_KEY")
        if not get_supabase_bucket():
            missing.append("SUPABASE_BUCKET")
        return missing
    missing = []
    missing.extend(["SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_BUCKET"])
    return missing


def _remember_upload(path: Path, backend: str) -> None:
    LAST_STATUS.update(
        {
            "configured": True,
            "backend": backend,
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


def _streamlit_secret_keys() -> tuple[list[str], str]:
    try:
        import streamlit as st

        return sorted([str(key) for key in st.secrets.keys()]), ""
    except Exception as exc:
        return [], str(exc)


def drive_configured() -> bool:
    return bool(get_drive_folder_id() and get_service_account_info())


def supabase_configured() -> bool:
    return bool(get_supabase_url() and get_supabase_key() and get_supabase_bucket())


def remote_storage_configured() -> bool:
    return supabase_configured() or drive_configured()


def active_backend() -> str:
    if supabase_configured():
        return "Supabase"
    if drive_configured():
        return "Google Drive"
    return ""


def get_supabase_url() -> str:
    return str(os.environ.get("SUPABASE_URL") or _streamlit_secret("SUPABASE_URL", "")).strip().rstrip("/")


def get_supabase_key() -> str:
    return str(
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or _streamlit_secret("SUPABASE_SERVICE_ROLE_KEY", "")
        or os.environ.get("SUPABASE_KEY")
        or _streamlit_secret("SUPABASE_KEY", "")
    ).strip()


def get_supabase_bucket() -> str:
    return str(os.environ.get("SUPABASE_BUCKET") or _streamlit_secret("SUPABASE_BUCKET", "")).strip()


def get_drive_folder_id() -> str:
    return str(os.environ.get("GOOGLE_DRIVE_FOLDER_ID") or _streamlit_secret("GOOGLE_DRIVE_FOLDER_ID", "")).strip()


def get_service_account_info() -> dict[str, Any]:
    for secret_name in SERVICE_ACCOUNT_JSON_SECRET_NAMES:
        raw_json = _streamlit_secret(secret_name, "")
        if not raw_json:
            continue
        try:
            return json.loads(str(raw_json), strict=False)
        except json.JSONDecodeError as exc:
            LAST_STATUS["last_error"] = f"{secret_name} is present but is not valid JSON: {exc}"
            return {}
    try:
        info = _streamlit_secret("gcp_service_account", {})
        if info:
            return dict(info)
    except Exception:
        pass
    return {}


def config_diagnostics() -> dict[str, Any]:
    keys, keys_error = _streamlit_secret_keys()
    service_table = _streamlit_secret("gcp_service_account", {})
    json_secret_names = [name for name in SERVICE_ACCOUNT_JSON_SECRET_NAMES if _streamlit_secret(name, "")]
    return {
        "active_backend": active_backend(),
        "supabase_url_present": bool(get_supabase_url()),
        "supabase_key_present": bool(get_supabase_key()),
        "supabase_bucket_present": bool(get_supabase_bucket()),
        "folder_id_present": bool(get_drive_folder_id()),
        "service_account_table_present": bool(service_table),
        "json_secret_names_present": json_secret_names,
        "top_level_secret_names": keys,
        "secret_read_error": keys_error,
        "last_error": LAST_STATUS.get("last_error", ""),
    }


def _supabase_object_url(path: Path) -> str:
    object_path = quote(relative_drive_path(path), safe="/")
    bucket = quote(get_supabase_bucket(), safe="")
    return f"{get_supabase_url()}/storage/v1/object/{bucket}/{object_path}"


def _supabase_headers() -> dict[str, str]:
    key = get_supabase_key()
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }


def upload_csv_to_supabase(path: Path) -> bool:
    if path.suffix.lower() != ".csv" or not path.exists() or not supabase_configured():
        return False
    headers = {
        **_supabase_headers(),
        "Content-Type": "text/csv",
        "x-upsert": "true",
    }
    response = requests.post(_supabase_object_url(path), headers=headers, data=path.read_bytes(), timeout=30)
    if response.status_code >= 400:
        raise RemoteStorageError(f"Supabase upload failed for {relative_drive_path(path)}: {response.status_code} {response.text}")
    _remember_upload(path, "Supabase")
    return True


def download_csv_from_supabase(path: Path) -> bool:
    if not supabase_configured():
        return False
    response = requests.get(_supabase_object_url(path), headers=_supabase_headers(), timeout=30)
    if response.status_code == 404:
        return False
    if response.status_code >= 400:
        raise RemoteStorageError(f"Supabase download failed for {relative_drive_path(path)}: {response.status_code} {response.text}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return True


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
        if supabase_configured():
            return upload_csv_to_supabase(path)
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
        _remember_upload(path, "Google Drive")
        return True
    except Exception as exc:
        _remember_error(exc)
        raise RemoteStorageError(f"Remote CSV upload failed for {relative_drive_path(path)}: {exc}") from exc


def download_csv(path: Path) -> bool:
    if supabase_configured():
        return download_csv_from_supabase(path)
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
    if not remote_storage_configured():
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
