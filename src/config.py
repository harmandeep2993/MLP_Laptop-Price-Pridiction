"""Configuration constants for the Laptop Price Prediction project."""

import os

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "laptop_data.csv")
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "cleaned_Data.csv")
PIPELINE_PATH = os.path.join(BASE_DIR, "pipe.pkl")
DATAFRAME_PATH = os.path.join(BASE_DIR, "df.pkl")

# ── Model Parameters ──────────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100

# ── Column Definitions ────────────────────────────────────────────────
CATEGORICAL_COLUMNS = ["Company", "OS", "Processor", "TypeName", "VC"]
CATEGORICAL_COLUMN_INDICES = [0, 1, 2, 3, 4]
TARGET_COLUMN = "Price"

FINAL_COLUMN_ORDER = [
    "Company", "OS", "Processor", "TypeName", "VC",
    "Ram", "TS", "Weight", "HDD", "SSD",
    "IPS", "Inches", "PPI", "Price",
]

# ── Feature Engineering Mappings ──────────────────────────────────────
HDD_REPLACEMENTS = {1: 1024, 2: 2048, 500: 512}
SSD_REPLACEMENTS = {500: 512, 1: 1024, 2: 2048, 240: 256, 180: 128}

SCREEN_SIZE_BINS = {
    (10.1, 11.3, 11.6, 12.0): 11.6,
    (12.3, 12.5, 13.0, 13.3, 13.5): 13.3,
    (13.9, 14.0, 14.1): 14.0,
    (15.0, 15.4, 15.6): 15.6,
}
SCREEN_SIZE_DEFAULT = 17.3
