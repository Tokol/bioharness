from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from harness_config import (
    BASE_FORMULATION_COLUMNS,
    FORMULATION_COMPONENTS,
    FORMULATION_COMPONENT_COLUMNS,
    FORMULATION_DATASET,
    FOR_TRAIN_DIR,
    LOCAL_DATASET_DIR,
    MATERIAL_LIBRARY,
    MATERIAL_LIBRARY_COLUMNS,
    MATERIAL_NAME_MAPPING,
    MATERIAL_NAME_MAPPING_COLUMNS,
    PROPERTY_TARGETS,
)
from utils import clean, contains_any, read_table, write_table


ENGINEERED_FEATURE_COLUMNS = [
    "target_value",
    "target_unit",
    "target_property_name",
    "coating_concentration_numeric",
    "feeding_rate_numeric",
    "has_fiber",
    "has_lignin",
    "has_coating",
    "has_plasticizer",
    "has_biopolymer_matrix",
    "has_pet_hdpe",
    "has_epoxy",
    "has_polyester_resin",
    "has_cellulose",
    "has_wood_fiber",
    "num_material_components",
    "fiber_to_matrix_ratio",
    "additive_to_matrix_ratio",
    "sisal_wt_percent",
    "sorghum_bicolor_wt_percent",
    "coconut_coir_wt_percent",
    "PALF_wt_percent",
    "coir_wt_percent",
    "wood_fiber_wt_percent",
    "hemp_fiber_wt_percent",
    "cellulose_fiber_wt_percent",
    "lignin_wt_percent",
    "PLA_wt_percent",
    "PHB_wt_percent",
    "PHBV_wt_percent",
    "epoxy_wt_percent",
    "polyester_resin_wt_percent",
    "PET_HDPE_wt_percent",
    "total_natural_fiber_wt_percent",
    "total_cellulose_based_wt_percent",
    "total_lignocellulosic_wt_percent",
    "total_biopolymer_matrix_wt_percent",
]


FEATURE_COLUMNS = BASE_FORMULATION_COLUMNS + ENGINEERED_FEATURE_COLUMNS


def empty_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=FEATURE_COLUMNS)


def extract_first_number(value: object) -> str:
    if pd.isna(value):
        return ""
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    return match.group(0) if match else ""


def numeric_from_text(value: object) -> float:
    found = extract_first_number(value)
    if not found:
        return np.nan
    return float(found)


def split_ids(value: object) -> list[str]:
    return [part.strip() for part in clean(value).split(";") if part.strip()]


def split_numbers(value: object) -> list[float]:
    out: list[float] = []
    for part in clean(value).split(";"):
        number = numeric_from_text(part)
        if not pd.isna(number):
            out.append(float(number))
    return out


def component_values(row: pd.Series, role: str) -> list[tuple[str, str, float]]:
    ids = split_ids(row.get(f"{role}_material_id"))
    names = [part.strip() for part in clean(row.get(f"{role}_material")).split(";") if part.strip()]
    values = split_numbers(row.get(f"{role}_wt_percent"))
    if not ids:
        return []
    if len(values) == len(ids):
        return [(ids[idx], names[idx] if idx < len(names) else ids[idx], values[idx]) for idx in range(len(ids))]
    single = numeric_from_text(row.get(f"{role}_wt_percent"))
    if not pd.isna(single):
        if len(ids) == 1:
            return [(ids[0], names[0] if names else ids[0], float(single))]
        return [(mid, names[idx] if idx < len(names) else mid, float(single) / len(ids)) for idx, mid in enumerate(ids)]
    return [(mid, names[idx] if idx < len(names) else mid, np.nan) for idx, mid in enumerate(ids)]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["coating_concentration", "feeding_rate"]:
        if col in out.columns:
            out[f"{col}_numeric"] = out[col].map(numeric_from_text)

    role_cols = ["matrix", "reinforcement", "additive", "coating", "plasticizer"]
    for idx, row in out.iterrows():
        components: list[tuple[str, str, float, str]] = []
        for role in role_cols:
            components.extend([(mid, name, pct, role) for mid, name, pct in component_values(row, role)])

        text = " ".join([name for _, name, _, _ in components]).lower()
        out.loc[idx, "has_fiber"] = int(any(role == "reinforcement" for _, _, _, role in components))
        out.loc[idx, "has_lignin"] = int("lignin" in text)
        out.loc[idx, "has_coating"] = int(any(role == "coating" for _, _, _, role in components))
        out.loc[idx, "has_plasticizer"] = int(any(role == "plasticizer" for _, _, _, role in components))
        out.loc[idx, "has_biopolymer_matrix"] = int(contains_any(text, ["pla", "phb", "phbv", "pha", "starch", "wood pulp", "cellulose"]))
        out.loc[idx, "has_pet_hdpe"] = int(contains_any(text, ["pet", "hdpe"]))
        out.loc[idx, "has_epoxy"] = int("epoxy" in text)
        out.loc[idx, "has_polyester_resin"] = int("polyester resin" in text)
        out.loc[idx, "has_cellulose"] = int("cellulose" in text or "wood pulp" in text)
        out.loc[idx, "has_wood_fiber"] = int(contains_any(text, ["wood fiber", "wood pulp", "thermomechanical pulp"]))
        out.loc[idx, "num_material_components"] = len(components)

        role_totals = {role: 0.0 for role in role_cols}
        for _, _, pct, role in components:
            if not pd.isna(pct):
                role_totals[role] += float(pct)
        matrix_total = role_totals["matrix"]
        out.loc[idx, "fiber_to_matrix_ratio"] = role_totals["reinforcement"] / matrix_total if matrix_total else np.nan
        out.loc[idx, "additive_to_matrix_ratio"] = role_totals["additive"] / matrix_total if matrix_total else np.nan

        specific = {
            "sisal_wt_percent": ["sisal"],
            "sorghum_bicolor_wt_percent": ["sorghum"],
            "coconut_coir_wt_percent": ["coconut coir"],
            "PALF_wt_percent": ["palf", "pineapple leaf"],
            "coir_wt_percent": ["coir"],
            "wood_fiber_wt_percent": ["wood fiber", "thermomechanical pulp", "wood pulp"],
            "hemp_fiber_wt_percent": ["hemp"],
            "cellulose_fiber_wt_percent": ["cellulose"],
            "lignin_wt_percent": ["lignin"],
            "PLA_wt_percent": ["pla"],
            "PHB_wt_percent": ["phb"],
            "PHBV_wt_percent": ["phbv"],
            "epoxy_wt_percent": ["epoxy"],
            "polyester_resin_wt_percent": ["polyester resin"],
            "PET_HDPE_wt_percent": ["pet", "hdpe"],
        }
        for col, terms in specific.items():
            total = 0.0
            for _, name, pct, _ in components:
                if not pd.isna(pct) and contains_any(name.lower(), terms):
                    total += float(pct)
            out.loc[idx, col] = total

        out.loc[idx, "total_natural_fiber_wt_percent"] = sum(
            float(out.loc[idx, col])
            for col in [
                "sisal_wt_percent",
                "sorghum_bicolor_wt_percent",
                "coconut_coir_wt_percent",
                "PALF_wt_percent",
                "coir_wt_percent",
                "wood_fiber_wt_percent",
                "hemp_fiber_wt_percent",
            ]
        )
        out.loc[idx, "total_cellulose_based_wt_percent"] = float(out.loc[idx, "wood_fiber_wt_percent"]) + float(out.loc[idx, "cellulose_fiber_wt_percent"])
        out.loc[idx, "total_lignocellulosic_wt_percent"] = float(out.loc[idx, "total_natural_fiber_wt_percent"]) + float(out.loc[idx, "lignin_wt_percent"])
        out.loc[idx, "total_biopolymer_matrix_wt_percent"] = float(out.loc[idx, "PLA_wt_percent"]) + float(out.loc[idx, "PHB_wt_percent"]) + float(out.loc[idx, "PHBV_wt_percent"])
    return out


def build_property_feature_dataset(formulations: pd.DataFrame, property_name: str) -> pd.DataFrame:
    if formulations.empty or "measured_property_name" not in formulations.columns:
        return empty_feature_frame()
    subset = formulations[formulations["measured_property_name"].fillna("") == property_name].copy()
    if subset.empty:
        return empty_feature_frame()
    subset["target_value"] = pd.to_numeric(subset["measured_property_value"], errors="coerce")
    subset["target_unit"] = subset["measured_property_unit"]
    subset["target_property_name"] = subset["measured_property_name"]
    subset = subset.dropna(subset=["target_value"])
    if subset.empty:
        return empty_feature_frame()
    subset = subset.sort_values(["formulation_id", "target_value"]).drop_duplicates("formulation_id", keep="first")
    return add_engineered_features(subset)


def ensure_csv_schema(path: Path, columns: list[str]) -> bool:
    existing = read_table(path)
    if existing.empty and not path.exists():
        write_table(pd.DataFrame(columns=columns), path)
        return True
    if existing.empty:
        missing = [col for col in columns if col not in existing.columns]
        if missing:
            write_table(pd.DataFrame(columns=columns), path)
            return True
        return False
    changed = False
    for col in columns:
        if col not in existing.columns:
            existing[col] = ""
            changed = True
    if changed:
        write_table(existing, path)
    return changed


def ensure_training_scaffold() -> dict[str, object]:
    LOCAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    FOR_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    created_or_updated: list[str] = []

    for path, columns in [
        (FORMULATION_DATASET, BASE_FORMULATION_COLUMNS),
        (FORMULATION_COMPONENTS, FORMULATION_COMPONENT_COLUMNS),
        (MATERIAL_LIBRARY, MATERIAL_LIBRARY_COLUMNS),
        (MATERIAL_NAME_MAPPING, MATERIAL_NAME_MAPPING_COLUMNS),
    ]:
        if ensure_csv_schema(path, columns):
            created_or_updated.append(str(path))

    return {
        "dataset_dir": LOCAL_DATASET_DIR,
        "training_dir": FOR_TRAIN_DIR,
        "created_or_updated": created_or_updated,
        "feature_files": [path for path in [FOR_TRAIN_DIR / filename for filename in PROPERTY_TARGETS.values()] if path.exists()],
    }


def regenerate_feature_csvs(material_project_dir: Path | None = None, training_dir: Path | None = None) -> dict[str, int]:
    formulation_path = FORMULATION_DATASET if material_project_dir is None else material_project_dir / "formulation_dataset.csv"
    train_dir = FOR_TRAIN_DIR if training_dir is None else training_dir
    train_dir.mkdir(parents=True, exist_ok=True)
    if not formulation_path.exists():
        formulations = pd.DataFrame(columns=BASE_FORMULATION_COLUMNS)
    else:
        formulations = pd.read_csv(formulation_path)
    result: dict[str, int] = {}
    for property_name, filename in PROPERTY_TARGETS.items():
        features = build_property_feature_dataset(formulations, property_name)
        for col in FEATURE_COLUMNS:
            if col not in features.columns:
                features[col] = ""
        features = features[FEATURE_COLUMNS]
        output_path = formulation_path.parent / filename
        write_table(features, output_path)
        write_table(features, train_dir / filename)
        result[filename] = len(features)
    return result


if __name__ == "__main__":
    counts = regenerate_feature_csvs()
    for name, count in counts.items():
        print(f"{name}: {count} rows")
