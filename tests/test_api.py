import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_endpoint():
    player_data = {
        "GP": 80,
        "MIN": 30.2,
        "PTS": 15.3,
        "FGM": 5.8,
        "FGA": 12.7,
        "FG_Percentage": 45.8,
        "ThreeP_Made": 1.2,
        "ThreePA": 3.6,
        "ThreeP_Percentage": 33.3,
        "FTM": 3.2,
        "FTA": 4.0,
        "FT_Percentage": 80.2,
        "OREB": 1.1,
        "DREB": 4.5,
        "REB": 5.6,
        "AST": 4.3,
        "STL": 1.2,
        "BLK": 0.5,
        "TOV": 2.1
    }

    response = client.post("/predict", data=player_data)
    assert response.status_code == 200
    assert "prediction" in response.text.lower()
