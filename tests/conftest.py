import os
import pytest
import joblib

@pytest.fixture(scope="session")
def model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "../app/model/rf_model.pkl")
    return joblib.load(MODEL_PATH)

