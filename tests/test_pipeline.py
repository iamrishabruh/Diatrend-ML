# tests/test_pipeline.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from pathlib import Path
from preprocessing.preprocessing import DataProcessor
from config.config import Config

def test_end_to_end():
    processor = DataProcessor()
    test_file = Path(Config.RAW_DATA) / "Subject1.xlsx"
    try:
        features, label = processor.process_file(test_file, processor._load_demographics())
        assert len(features) == Config.NUM_FEATURES
        assert label in [0, 1]
    except Exception as e:
        pytest.fail(f"Pipeline failed: {str(e)}")
