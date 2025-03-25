# preprocessing/preprocessing.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List, Union
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel, ValidationError, Field, validator, ConfigDict
from pydantic_core import CoreSchema, core_schema
from config.config import Config

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.features = []
        self.labels = []

    def _load_demographics(self) -> pd.DataFrame:
        """Load demographics data with robust whitespace stripping and numeric conversion."""
        df = pd.read_excel(Config.DEMOGRAPHICS_PATH)
        df = df.copy()
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()
        df["Hemoglobin A1C"] = pd.to_numeric(df["Hemoglobin A1C"], errors="coerce")
        df["Hemoglobin A1C"] = df["Hemoglobin A1C"].fillna(df["Hemoglobin A1C"].mean())
        # Convert Age: extract numeric portion (lower bound) using regex
        df["Age"] = df["Age"].astype(str).str.strip().str.extract(r"(\d+\.?\d*)")[0]
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df["Age"] = df["Age"].fillna(df["Age"].mean())
        return df

    def _parse_age(self, age: str) -> float:
        """Convert age ranges (e.g., '35 - 44 yrs') to numeric (e.g., 35.0)."""
        try:
            if pd.isna(age) or age == "":
                return np.nan
            age = age.strip()
            if "-" in age:
                return float(age.split("-")[0].strip())
            return float(age)
        except Exception as e:
            logger.error(f"Failed to convert age: {age} - {str(e)}")
            return np.nan

    def process_file(self, file_input: Union[Path, BytesIO], demo_df: pd.DataFrame, filename: str = None) -> Tuple[List[float], int]:
        """
        Process a subject file (Path or BytesIO) from the CGM sheet.
        Strips extra whitespace, extracts numeric values using regex, drops rows if conversion fails,
        and returns a feature vector and label.
        """
        try:
            # Determine subject ID
            if isinstance(file_input, BytesIO):
                if filename is None:
                    raise ValueError("Filename must be provided for BytesIO input.")
                subject_id = int(filename.replace("Subject", "").replace(".xlsx", "").strip())
                file_buffer = file_input
            else:
                subject_id = int(Path(file_input).stem.replace("Subject", "").strip())
                file_buffer = file_input

            # Load CGM data
            cgm_df = pd.read_excel(file_buffer, sheet_name="CGM")
            cgm_df = cgm_df.copy()
            cgm_df.columns = cgm_df.columns.str.strip()
            if "mg/dl" not in cgm_df.columns:
                raise ValueError("Column 'mg/dl' not found in CGM sheet.")
            # Convert "mg/dl": extract numeric portion using regex
            cgm_df["mg/dl"] = cgm_df["mg/dl"].astype(str).str.strip().str.extract(r"(\d+\.?\d*)")[0]
            cgm_df["mg/dl"] = pd.to_numeric(cgm_df["mg/dl"], errors="coerce")
            cgm_df = cgm_df.dropna(subset=["mg/dl"])
            cgm_df["mg/dl"] = cgm_df["mg/dl"].fillna(cgm_df["mg/dl"].mean())

            # Extract features from CGM: mean, std, and glycemic variability
            cgm_values = cgm_df["mg/dl"].values
            cgm_features = [
                np.mean(cgm_values),
                np.std(cgm_values),
                np.sqrt(np.mean(np.diff(cgm_values)**2)) if len(cgm_values) > 1 else 0.0
            ]

            # Process demographics: ensure headers and cells are stripped
            demo_df = demo_df.copy()
            demo_df.columns = demo_df.columns.str.strip()
            for col in demo_df.select_dtypes(include="object").columns:
                demo_df[col] = demo_df[col].str.strip()
            demo_row = demo_df[demo_df["Subject"] == subject_id]
            if demo_row.empty:
                raise ValueError(f"No demographics found for subject {subject_id}")
            # Extract Age using regex
            age_str = demo_row["Age"].values[0]
            age_extracted = pd.Series(str(age_str)).str.strip().str.extract(r"(\d+\.?\d*)")[0]
            age = pd.to_numeric(age_extracted, errors="coerce").iloc[0]
            hba1c = pd.to_numeric(demo_row["Hemoglobin A1C"].values[0], errors="coerce")
            if pd.isna(age):
                age = 0.0
            if pd.isna(hba1c):
                hba1c = 0.0
            demo_features = [age, hba1c]

            # Label: 0 if HbA1C > 7.0, else 1
            label = 0 if hba1c > 7.0 else 1

            features_combined = cgm_features + demo_features
            features_combined = [float(x) if (x is not None and not np.isnan(x) and not np.isinf(x)) else 0.0 for x in features_combined]
            return features_combined, label

        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise

    def load_dataset(self):
        """Load dataset from all Subject*.xlsx files and normalize features."""
        demo_df = self._load_demographics()
        for file in Config.RAW_DATA.glob("Subject*.xlsx"):
            try:
                features, label = self.process_file(file, demo_df)
                if not np.isnan(features).any() and not np.isinf(features).any():
                    self.features.append(features)
                    self.labels.append(label)
                else:
                    logger.error(f"File {file.name} contains NaN or Inf in features.")
            except Exception as e:
                logger.error(f"Skipping {file.name}: {str(e)}")
                continue
        features_arr = np.array(self.features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_arr)
        return features_scaled, np.array(self.labels)

