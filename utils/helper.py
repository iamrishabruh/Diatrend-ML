# utils/helper.py
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def load_data(file_path):
    """
    Loads a CSV or Excel file into a Pandas DataFrame.
    """
    _, ext = os.path.splitext(file_path)
    try:
        if ext.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif ext.lower() in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file extension: " + ext)
        logger.info("Data loaded successfully from %s", file_path)
        return df
    except Exception as e:
        logger.error("Error loading data from %s: %s", file_path, e)
        raise
