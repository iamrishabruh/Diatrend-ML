# preprocessing/preprocessing.py

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List, Union
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from config.config import Config

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.features = []
        self.labels = []

    def _load_demographics(self) -> pd.DataFrame:
        """Load and validate demographics data, stripping extra whitespace and converting Age."""
        df = pd.read_excel(Config.DEMOGRAPHICS_PATH)
        df = df.copy()
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()

        df["Hemoglobin A1C"] = pd.to_numeric(df["Hemoglobin A1C"], errors="coerce")
        df["Hemoglobin A1C"] = df["Hemoglobin A1C"].fillna(df["Hemoglobin A1C"].mean())
        df["Age"] = df["Age"].astype(str).str.strip().str.extract(r"(\d+\.?\d*)")[0]
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df["Age"] = df["Age"].fillna(df["Age"].mean())
        return df

    def process_file(self, file_input: Union[Path, BytesIO], demo_df: pd.DataFrame, filename: str = None) -> Tuple[List[float], int]:
        try:
            if isinstance(file_input, BytesIO):
                if filename is None:
                    raise ValueError("Filename must be provided for BytesIO input.")
                subject_id = int(filename.replace("Subject", "").replace(".xlsx", "").strip())
                file_buffer = file_input
            else:
                subject_id = int(Path(file_input).stem.replace("Subject", "").strip())
                file_buffer = file_input

            cgm_df = pd.read_excel(file_buffer, sheet_name="CGM")
            cgm_df = cgm_df.copy()
            cgm_df.columns = cgm_df.columns.str.strip()
            if "mg/dl" not in cgm_df.columns:
                raise ValueError("Column 'mg/dl' not found in CGM sheet.")

            cgm_df["mg/dl"] = cgm_df["mg/dl"].astype(str).str.strip().str.extract(r"(\d+\.?\d*)")[0]
            cgm_df["mg/dl"] = pd.to_numeric(cgm_df["mg/dl"], errors="coerce")
            cgm_df = cgm_df.dropna(subset=["mg/dl"])
            cgm_df["mg/dl"] = cgm_df["mg/dl"].fillna(cgm_df["mg/dl"].mean())

            cgm_values = cgm_df["mg/dl"].values
            cgm_features = [
                np.mean(cgm_values),
                np.std(cgm_values),
                np.sqrt(np.mean(np.diff(cgm_values)**2)) if len(cgm_values) > 1 else 0.0
            ]

            demo_df = demo_df.copy()
            demo_df.columns = demo_df.columns.str.strip()
            for col in demo_df.select_dtypes(include="object").columns:
                demo_df[col] = demo_df[col].str.strip()

            demo_row = demo_df[demo_df["Subject"] == subject_id]
            if demo_row.empty:
                raise ValueError(f"No demographics found for subject {subject_id}")

            age_str = demo_row["Age"].values[0]
            age_extracted = pd.Series(str(age_str)).str.strip().str.extract(r"(\d+\.?\d*)")[0]
            age = pd.to_numeric(age_extracted, errors="coerce").iloc[0]
            hba1c = pd.to_numeric(demo_row["Hemoglobin A1C"].values[0], errors="coerce")

            if pd.isna(age):
                age = 0.0
            if pd.isna(hba1c):
                hba1c = 0.0
            demo_features = [age, hba1c]

            label = 0 if hba1c > 7.0 else 1

            features_combined = cgm_features + demo_features
            features_combined = [float(x) if (x is not None and not np.isnan(x) and not np.isinf(x)) else 0.0 for x in features_combined]
            return features_combined, label

        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise

    def _oversample_knn(self, X: np.ndarray, y: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple k-NN interpolation to oversample the minority class.
        """
        classes, counts = np.unique(y, return_counts=True)
        max_count = counts.max()
        X_aug, y_aug = [X], [y]
        nbr = NearestNeighbors(n_neighbors=k+1).fit(X)
        for cls, count in zip(classes, counts):
            if count < max_count:
                X_cls = X[y == cls]
                needed = max_count - count
                for _ in range(needed):
                    idx = np.random.randint(len(X_cls))
                    # get k neighbors
                    _, neighbor_idx = nbr.kneighbors(X_cls[idx].reshape(1, -1))
                    neighbor_pool = neighbor_idx[0][1:]
                    chosen_neighbor = X[neighbor_pool[np.random.randint(len(neighbor_pool))]]
                    alpha = np.random.rand()
                    new_sample = X_cls[idx] + alpha * (chosen_neighbor - X_cls[idx])
                    X_aug.append(new_sample.reshape(1, -1))
                    y_aug.append(np.array([cls]))
        return np.vstack(X_aug), np.concatenate(y_aug)

    def load_dataset(self, augment: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the entire dataset of 55 subjects. If augment=True, produce an oversampled set
        for training while also returning the original data for validation.
        """
        demo_df = self._load_demographics()
        self.features.clear()
        self.labels.clear()

        for file in Config.RAW_DATA.glob("Subject*.xlsx"):
            try:
                feats, lbl = self.process_file(file, demo_df)
                self.features.append(feats)
                self.labels.append(lbl)
            except Exception as e:
                logger.error(f"Skipping {file.name}: {str(e)}")

        X = np.array(self.features)
        y = np.array(self.labels)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if augment:
            X_aug, y_aug = self._oversample_knn(X_scaled, y, k=3)
            return X_aug, y_aug  # Return the augmented data for training
        else:
            return X_scaled, y
