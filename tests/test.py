import requests

url = "http://127.0.0.1:8000/predict"
player_data = {
    "MIN": 30.2,
    "PTS": 15.3,
    "FGM": 5.8,
    "FGA": 12.7,
    "ThreeP_Made": 1.2,
    "ThreePA": 3.6,
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

response = requests.post(url, json=player_data)
print(response.json()) 
