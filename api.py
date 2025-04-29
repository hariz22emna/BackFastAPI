from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List
from sqlalchemy import create_engine

# =============================
# 🔹 Chargement des modèles
# =============================
model_wait = joblib.load("model_xgb.pkl")                     # Modèle temps d’attente
model_surcharge = joblib.load("gradient_boosting_model.pkl")  # Modèle surcharge
encoders = joblib.load("label_encoders.pkl")                  # LabelEncoders pour catégorielles

# =============================
# 🔹 Connexion MySQL
# =============================
DB_URL = "mysql+mysqlconnector://root:@localhost/pfadataset"
engine = create_engine(DB_URL)

# =============================
# 🔹 Colonnes d’entrée (temps d’attente)
# =============================
feature_names = [
    "urgency_level",
    "nurse-to-patient_ratio",
    "specialist_availability",
    "time_to_registration_min",
    "time_to_medical_professional_min",
    "available_beds_%"
]

# =============================
# 🔹 Initialisation FastAPI
# =============================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:53440"],  # Adresse front-end
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# 🔹 Modèles d’entrée
# =============================
class InputData(BaseModel):
    features: List[float]

class BatchInput(BaseModel):
    features_batch: List[List[float]]

class SurchargeInput(BaseModel):
    visit_date: str
    day_of_week: str
    available_beds: float
    urgency_level: str
    season: str
    local_event: int
    nurse_to_patient_ratio: int

# =============================
# 🔹 Home route
# =============================
@app.get("/")
def home():
    return {"message": "🎯 API FastAPI pour prédiction du temps d’attente et surcharge"}

# =============================
# 🔹 /predict : prédiction simple (temps d’attente)
# =============================
@app.post("/predict")
def predict(data: InputData):
    if len(data.features) != len(feature_names):
        return {
            "error": f"Expected {len(feature_names)} features, got {len(data.features)}",
            "expected_order": feature_names
        }
    input_array = np.array(data.features).reshape(1, -1)
    prediction = model_wait.predict(input_array)
    return {"prediction": float(prediction[0])}

# =============================
# 🔹 /batch_predict : prédictions multiples
# =============================
@app.post("/batch_predict")
def batch_predict(data: BatchInput):
    batch = np.array(data.features_batch)
    if batch.shape[1] != len(feature_names):
        return {
            "error": f"Each row must have {len(feature_names)} features, got {batch.shape[1]}",
            "expected_order": feature_names
        }
    predictions = model_wait.predict(batch)
    return {"predictions": predictions.tolist()}

# =============================
# 🔹 /predict_from_db
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
    predictions = model_wait.predict(X)

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
# 🔹 /predict_with_real
# =============================
@app.post("/predict_with_real")
def predict_with_real(data: InputData):
    input_array = np.array(data.features).reshape(1, -1)
    prediction = model_wait.predict(input_array)[0]

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

# =============================
# 🔹 /predict_surcharge : prédiction surcharge Oui/Non
# =============================
@app.post("/predict_surcharge")
def predict_surcharge(data: SurchargeInput):
    try:
        # Encodage des variables catégorielles
        day_of_week_encoded = encoders["Day of Week"].transform([data.day_of_week])[0]
        urgency_level_encoded = encoders["Urgency Level"].transform([data.urgency_level])[0]
        season_encoded = encoders["Season"].transform([data.season])[0]

        # Conversion de la date
        visit_timestamp = pd.to_datetime(data.visit_date).timestamp()

        # Construction du vecteur d'entrée
        input_data = np.array([
            visit_timestamp,
            day_of_week_encoded,
            data.available_beds,
            urgency_level_encoded,
            season_encoded,
            data.local_event,
            data.nurse_to_patient_ratio
        ]).reshape(1, -1)

        # Prédiction
        prediction = model_surcharge.predict(input_data)[0]

        return {
            "prediction": int(prediction),
            "surcharge": "Oui" if prediction == 1 else "Non"
        }

    except Exception as e:
        return {"error": str(e)}
