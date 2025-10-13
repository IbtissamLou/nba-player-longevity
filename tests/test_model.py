import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../app/model/rf_model.pkl")

def test_model_exists():
    assert os.path.exists(MODEL_PATH), "Model file not found!"

def test_model_loads():
    model = joblib.load(MODEL_PATH)
    assert model is not None, "Failed to load model!"

def test_model_prediction_shape_and_value():
    model = joblib.load(MODEL_PATH)

    # Create fake input (same number of features as your app expects)
    sample = np.array([[80, 30.2, 15.3, 5.8, 12.7, 45.8, 1.2, 3.6, 33.3,
                        3.2, 4.0, 80.2, 1.1, 4.5, 5.6, 4.3, 1.2, 0.5, 2.1]])
    
    pred = model.predict(sample)
    prob = model.predict_proba(sample)[:, 1]

    assert pred.shape == (1,)
    assert 0 <= prob[0] <= 1
