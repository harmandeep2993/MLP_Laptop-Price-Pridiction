"""Feature engineering transformations for the laptop dataset."""

import re

import numpy as np
import pandas as pd

from .config import (
    FINAL_COLUMN_ORDER,
    HDD_REPLACEMENTS,
    SCREEN_SIZE_BINS,
    SCREEN_SIZE_DEFAULT,
    SSD_REPLACEMENTS,
)


# ── Helper functions ──────────────────────────────────────────────────

def _convert_screen_size(size: float) -> float:
    """Bin raw screen sizes into standard categories."""
    for sizes, replacement in SCREEN_SIZE_BINS.items():
        if size in sizes:
            return replacement
    return SCREEN_SIZE_DEFAULT


def _extract_processor(text: str) -> str:
    """Categorize a CPU string into a high-level processor group."""
    if text in ("Intel Core i7", "Intel Core i5", "Intel Core i3"):
        return text
    if text.split()[0] == "Intel":
        return "Other Intel Processor"
    return "AMD Processor"


def _extract_opsys(text: str) -> str:
    """Map raw OS names to simplified categories."""
    if text in ("Windows 10", "Windows 10 S", "Windows 7"):
        return "WindowsOS"
    if text == "Linux":
        return "LinuxOS"
    if text in ("macOS", "Mac OS X"):
        return "MacOS"
    return "OtherOS"


# ── Main orchestrator ─────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps and return the final DataFrame."""
    df = df.copy()

    # RAM – remove 'GB', cast to int
    df["Ram"] = df["Ram"].str.replace("GB", "").astype(int)

    # Weight – remove 'kg', cast to float
    df["Weight"] = df["Weight"].str.replace("kg", "").astype(float)

    # ScreenResolution → Touchscreen, IPS, PPI
    df["TS"] = df["ScreenResolution"].apply(lambda x: 1 if "Touchscreen" in x else 0)
    df["IPS"] = df["ScreenResolution"].apply(lambda x: 1 if "IPS" in x else 0)

    # Screen size binning
    df["Inches"] = df["Inches"].apply(_convert_screen_size)

    # Resolution → PPI
    resolution = df["ScreenResolution"].str.extract(r"(\d+)x(\d+)", expand=True).astype(int)
    df["Width_pixels"] = resolution[0]
    df["Height_pixels"] = resolution[1]
    df["PPI"] = round(
        ((df["Width_pixels"] ** 2 + df["Height_pixels"] ** 2) ** 0.5) / df["Inches"]
    ).astype(float)
    df = df.drop(columns=["ScreenResolution", "Width_pixels", "Height_pixels"])

    # TypeName
    df["TypeName"] = df["TypeName"].replace(
        {"2 in 1 Convertible": "Convertible", "Netbook": "Notebook"}
    )

    # Processor
    df["Processor"] = df["Cpu"].apply(lambda x: " ".join(x.split()[0:3]))
    df["Processor"] = df["Processor"].apply(_extract_processor)
    df = df.drop(columns=["Cpu"])

    # Memory → HDD, SSD, FS, HBD
    patterns = {
        "SSD": r"(\d+(?:\.\d+)?(?:GB|TB))\s+SSD",
        "HDD": r"(\d+(?:\.\d+)?(?:TB|GB))\s+HDD",
        "FS": r"(\d+GB)\s+Flash\s+Storage",
        "HBD": r"(\d+(?:\.\d+)?(?:GB|TB))\s+Hybrid",
    }
    for col in ("SSD", "HDD", "FS", "HBD"):
        df[col] = "0"
    for pattern_type, pattern in patterns.items():
        for idx, row in df.iterrows():
            match = re.search(pattern, row["Memory"])
            if match:
                df.at[idx, pattern_type] = match.group(1)

    for col in ("SSD", "HDD", "FS", "HBD"):
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("GB", "")
            .str.replace("TB", "")
        )
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).round().astype(int)

    df = df.drop(columns=["Memory"])

    # Operating System
    df["OS"] = df["OpSys"].apply(_extract_opsys)
    df = df.drop(columns=["OpSys"])

    # Video Card
    df["VC"] = df["Gpu"].apply(lambda x: x.split()[0])
    df = df.drop(columns=["Gpu"])

    # Drop Flash Storage & Hybrid (low variance)
    df = df.drop(columns=["FS", "HBD"], errors="ignore")

    # Replace storage values with standard capacities
    df["HDD"] = df["HDD"].replace(HDD_REPLACEMENTS)
    df["SSD"] = df["SSD"].replace(SSD_REPLACEMENTS)

    # Final column order
    df = df[FINAL_COLUMN_ORDER]

    return df
