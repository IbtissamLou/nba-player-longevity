from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
import numpy as np
import joblib
import os 

# importer le modèle enregistré
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR,"model", "rf_model.pkl")

model = joblib.load(MODEL_PATH)

# Initialiser FastAPI
app = FastAPI(title="NBA Player Career Prediction API",
              description="API REST permettant de prédire si un joueur NBA aura une carrière de plus de 5 ans.",
              version="1.0")

# Configurer template Jinja2 
TEM_PATH = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEM_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def home(request: Request):  
    return templates.TemplateResponse("index.html", {"request": request})  

@app.post("/predict")
def predict(
    request: Request,  
    GP: float = Form(...),
    MIN: float = Form(...),
    PTS: float = Form(...),
    FGM: float = Form(...),
    FGA: float = Form(...),
    FG_Percentage: float = Form(...),
    ThreeP_Made: float = Form(...),
    ThreePA: float = Form(...),
    ThreeP_Percentage: float = Form(...),
    FTM: float = Form(...),
    FTA: float = Form(...),
    FT_Percentage: float = Form(...),
    OREB: float = Form(...),
    DREB: float = Form(...),
    REB: float = Form(...),
    AST: float = Form(...),
    STL: float = Form(...),
    BLK: float = Form(...),
    TOV: float = Form(...)
):
    """
    Prédire si un joueur NBA aura une carrière plus que 5 ans.
    """
    # Convert input to numpy array
    player_data = np.array([[
        GP, MIN, PTS, FGM, FGA, FG_Percentage,
        ThreeP_Made, ThreePA, ThreeP_Percentage, 
        FTM, FTA, FT_Percentage, 
        OREB, DREB, REB, AST, STL, BLK, TOV
    ]])

    # Make prediction
    prediction = model.predict(player_data)
    prediction_proba = model.predict_proba(player_data)[:, 1]

    result = "Long Career (5+ years)" if prediction[0] == 1 else "Short Career (<5 years)"

    return templates.TemplateResponse("result.html", {
        "request": request,  
        "prediction": result,
        "probability": round(float(prediction_proba[0]), 4)
})



