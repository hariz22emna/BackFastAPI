from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List
from sqlalchemy import create_engine
import pandas as pd

# Charger le modèle
model = joblib.load("model_xgb.pkl")

# Connexion MySQL
DB_URL = "mysql+mysqlconnector://root:@localhost/pfadataset"
engine = create_engine(DB_URL)

# Colonnes attendues
feature_names = [
    "urgency_level",
    "nurse-to-patient_ratio",
    "specialist_availability",
    "time_to_registration_min",
    "time_to_medical_professional_min",
    "available_beds_%"
]

# Initialiser l'app FastAPI
app = FastAPI()

# Configuration CORS (pour Angular)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:53440"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# 🎯 Input model
# =============================
class InputData(BaseModel):
    features: List[float]

class BatchInput(BaseModel):
    features_batch: List[List[float]]

# =============================
# 🔹 /predict : prédiction simple
# =============================
@app.post("/predict")
def predict(data: InputData):
    if len(data.features) != len(feature_names):
        return {
            "error": f"Expected {len(feature_names)} features, got {len(data.features)}",
            "expected_order": feature_names
        }
    input_array = np.array(data.features).reshape(1, -1)
    prediction = model.predict(input_array)
    return {"prediction": float(prediction[0])}

# =============================
# 🔹 /batch_predict : plusieurs prédictions
# =============================
@app.post("/batch_predict")
def batch_predict(data: BatchInput):
    batch = np.array(data.features_batch)
    if batch.shape[1] != len(feature_names):
        return {
            "error": f"Each row must have {len(feature_names)} features, got {batch.shape[1]}",
            "expected_order": feature_names
        }
    predictions = model.predict(batch)
    return {"predictions": predictions.tolist()}

# =============================
# 🔹 /predict_from_db : comparer 100 prédictions avec données réelles
# =============================
@app.get("/predict_from_db")
def predict_from_db(limit: int = 100):
    query = f"""
        SELECT
            urgency_level,
            `nurse-to-patient_ratio`,
            specialist_availability,
            time_to_registration_min,
            time_to_medical_professional_min,
            `available_beds_%`,
            total_wait_time_min
        FROM donnees_pretraitees
        LIMIT {limit}
    """
    df = pd.read_sql(query, con=engine)

    if df.empty:
        return {"error": "No data available."}

    X = df[feature_names]
    y_real = df["total_wait_time_min"]
    predictions = model.predict(X)

    results = []
    for pred, real in zip(predictions, y_real):
        results.append({
            "prediction": float(pred),
            "real_value": float(real)
        })

    return {
        "count": len(results),
        "results": results
    }

# =============================
# 🔹 /predict_with_real : prédiction + valeur réelle correspondante
# =============================
@app.post("/predict_with_real")
def predict_with_real(data: InputData):
    input_array = np.array(data.features).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    # Construire la requête SQL pour chercher une ligne identique
    query = f"""
        SELECT total_wait_time_min FROM donnees_pretraitees
        WHERE urgency_level = {data.features[0]}
          AND `nurse-to-patient_ratio` = {data.features[1]}
          AND specialist_availability = {data.features[2]}
          AND time_to_registration_min = {data.features[3]}
          AND time_to_medical_professional_min = {data.features[4]}
          AND `available_beds_%` = {data.features[5]}
        LIMIT 1
    """
    df = pd.read_sql(query, engine)
    real = float(df["total_wait_time_min"][0]) if not df.empty else None

    return {
        "prediction": float(prediction),
        "real_value": real
    }
