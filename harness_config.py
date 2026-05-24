from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
REVIEW_DIR = DATA_DIR / "review"
APPLIED_DIR = DATA_DIR / "applied_reviews"
VALIDATION_DIR = DATA_DIR / "validation"
EXTRACTION_DIR = DATA_DIR / "extraction"
LOG_DIR = DATA_DIR / "logs"
LOCAL_DATASET_DIR = DATA_DIR / "datasets"
APPLIED_LOG = LOG_DIR / "applied_reviews.csv"
EXTRACTED_SOURCE_XLSX = EXTRACTION_DIR / "biomaterial_source_papers.xlsx"
EXTRACTED_SOURCE_CSV = EXTRACTION_DIR / "biomaterial_source_papers.csv"
VALIDATION_LOG = VALIDATION_DIR / "validation_log.csv"
REJECTION_LOG = VALIDATION_DIR / "rejected_papers.csv"
FOR_TRAIN_DIR = BASE_DIR / "forTrain"
OPENAI_KEY_FILE = BASE_DIR / "openai_key.txt"

MATERIAL_PROJECT_DIR = LOCAL_DATASET_DIR

SOURCE_TABLE_CSV = EXTRACTED_SOURCE_CSV
SOURCE_TABLE_XLSX = EXTRACTED_SOURCE_XLSX
FORMULATION_DATASET = LOCAL_DATASET_DIR / "formulation_dataset.csv"
FORMULATION_COMPONENTS = LOCAL_DATASET_DIR / "formulation_components.csv"
MATERIAL_LIBRARY = LOCAL_DATASET_DIR / "material_library.csv"
MATERIAL_NAME_MAPPING = LOCAL_DATASET_DIR / "material_name_mapping.csv"
MATERIAL_COST = LOCAL_DATASET_DIR / "material_cost_static.csv"
MATERIAL_CARBON = LOCAL_DATASET_DIR / "material_carbon_static.csv"

PROPERTY_TARGETS = {
    "tensile_strength": "tensile_strength_model_dataset_features.csv",
    "water_absorption": "water_absorption_model_dataset_features.csv",
    "moisture_absorption": "moisture_absorption_model_dataset_features.csv",
    "flexural_strength": "flexural_strength_model_dataset_features.csv",
    "impact_strength": "impact_strength_model_dataset_features.csv",
    "hardness": "hardness_model_dataset_features.csv",
    "storage_modulus": "storage_modulus_model_dataset_features.csv",
    "tear_index": "tear_index_model_dataset_features.csv",
    "burst_index": "burst_index_model_dataset_features.csv",
}

SOURCE_RECORD_COLUMNS = [
    "Validation Status",
    "Confidence Score",
    "Paper Quality Score",
    "Title",
    "Authors",
    "URL",
    "DOI",
    "Identity Status",
    "Published Date",
    "Available Inputs",
    "Available Outputs",
    "Has Extractable Table",
    "Material System",
    "Notes",
    "Number Of Possible Rows",
    "Paper Title",
    "Source Link",
    "Target Application",
    "Summary",
    "Conclusion",
    "Evidence",
    "Accees",
]

BASE_FORMULATION_COLUMNS = [
    "formulation_id",
    "source_paper_id",
    "source_paper_title",
    "application_domain",
    "material_system",
    "formulation_code",
    "matrix_material",
    "matrix_material_id",
    "reinforcement_material",
    "reinforcement_material_id",
    "additive_material",
    "additive_material_id",
    "coating_material",
    "coating_material_id",
    "plasticizer_material",
    "plasticizer_material_id",
    "matrix_wt_percent",
    "reinforcement_wt_percent",
    "additive_wt_percent",
    "coating_concentration",
    "plasticizer_wt_percent",
    "processing_method",
    "processing_temperature_c",
    "pressure_mpa",
    "processing_time_min",
    "screw_speed_rpm",
    "feeding_rate",
    "measured_property_name",
    "measured_property_value",
    "measured_property_unit",
    "test_standard",
    "property_group",
    "confidence_level",
    "notes",
]

MATERIAL_LIBRARY_COLUMNS = [
    "material_id",
    "material_name",
    "material_category",
    "material_family",
    "role_in_formulation",
    "bio_based_origin",
    "source_type",
    "source_paper_id",
    "source_paper_title",
    "grade_or_supplier",
    "known_density_g_cm3",
    "known_processing_temp_c",
    "biodegradability_note",
    "typical_application",
    "confidence_level",
    "notes",
]

FORMULATION_COMPONENT_COLUMNS = [
    "component_id",
    "formulation_id",
    "source_paper_id",
    "source_paper_title",
    "formulation_code",
    "material_id",
    "material_name",
    "canonical_material_name",
    "role",
    "wt_percent",
    "concentration",
    "is_primary_matrix",
    "source_section",
    "source_table_or_figure",
    "source_evidence",
    "confidence_level",
    "mapping_status",
    "notes",
]

MATERIAL_NAME_MAPPING_COLUMNS = [
    "raw_name",
    "canonical_material_name",
    "material_id",
    "notes",
]
