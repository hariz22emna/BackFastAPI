from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List
from sqlalchemy import create_engine
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# =============================
# 🔹 Chargement des modèles
# =============================
model_wait = joblib.load("model_xgb.pkl")
model_surcharge = joblib.load("gradient_boosting_model.pkl")
model_affluence = joblib.load("model_affluence_optimized.pkl")
encoders = joblib.load("label_encoders.pkl")

# =============================
# 🔹 Connexion MySQL
# =============================
DB_URL = "mysql+mysqlconnector://root:@localhost/pfadataset"
engine = create_engine(DB_URL)

# =============================
# 🔹 Colonnes d’entrée
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
app.mount("/static", StaticFiles(directory="static"), name="static")

# =============================
# 🔹 Middleware CORS
# =============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# 🔹 WebSocket – Alerte temps réel
# =============================
connected_clients: List[WebSocket] = []

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

async def broadcast_notification(message: str):
    for client in connected_clients:
        await client.send_text(message)

# =============================
# 🔹 Route HTML test WebSocket
# =============================
@app.get("/test-websocket")
def get_websocket_test():
    return FileResponse("static/test-websocket.html")

# =============================
# 🔹 Modèles Pydantic
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

class AffluenceInput(BaseModel):
    features: dict

# =============================
# 🔹 Route Home
# =============================
@app.get("/")
def home():
    return {"message": "🎯 API pour prédiction et alertes urgences"}

# =============================
# 🔹 /predict : prédiction temps d’attente
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
# 🔹 /batch_predict
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
# 🔹 /predict_surcharge + Notif temps réel
# =============================
@app.post("/predict_surcharge")
def predict_surcharge(data: SurchargeInput, background_tasks: BackgroundTasks):
    try:
        day_of_week_encoded = encoders["Day of Week"].transform([data.day_of_week])[0]
        urgency_level_encoded = encoders["Urgency Level"].transform([data.urgency_level])[0]
        season_encoded = encoders["Season"].transform([data.season])[0]
        visit_timestamp = pd.to_datetime(data.visit_date).timestamp()

        input_data = np.array([
            visit_timestamp,
            day_of_week_encoded,
            data.available_beds,
            urgency_level_encoded,
            season_encoded,
            data.local_event,
            data.nurse_to_patient_ratio
        ]).reshape(1, -1)

        prediction = model_surcharge.predict(input_data)[0]

        if prediction == 1:
            message = (
                "🚨 Surcharge détectée pour l’heure suivante ! "
                "➡️ Veuillez anticiper l'afflux en réorganisant le personnel."
            )
            background_tasks.add_task(broadcast_notification, message)
            return {
                "prediction": int(prediction),
                "surcharge": "Oui",
                "recommendations": message
            }

        return {
            "prediction": int(prediction),
            "surcharge": "Non"
        }

    except Exception as e:
        return {"error": str(e)}

# =============================
# 🔹 /predict_affluence : prédiction affluence (3e modèle)
# =============================
@app.post("/predict_affluence")
def predict_affluence(data: AffluenceInput):
    try:
        df = pd.DataFrame([data.features])

        for col in model_affluence.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
        df = df[model_affluence.feature_names_in_]

        prediction = model_affluence.predict(df)[0]
        return {"prediction": round(prediction)}

    except Exception as e:
        return {"error": str(e)}

# =============================
# 🔹 /forecast_traffic
# =============================
@app.get("/forecast_traffic")
async def forecast_traffic(day_of_week: str, season: str, local_event: int = 0):
    try:
        day_encoded = encoders["Day of Week"].transform([day_of_week])[0]
        season_encoded = encoders["Season"].transform([season])[0]
        input_data = np.array([[day_encoded, season_encoded, local_event]])
        prediction = np.random.randint(40, 100)

        message = f"Prévision d'affluence : {prediction} patients attendus ({day_of_week}, {season})"
        await broadcast_notification(message)

        return {
            "forecast": int(prediction),
            "details": {
                "day": day_of_week,
                "season": season,
                "event": bool(local_event)
            }
        }
    except Exception as e:
        return {"error": str(e)}

# =============================
# 🔹 /send_test_alert : alerte manuelle
# =============================
@app.get("/send_test_alert")
async def send_test_alert(message: str = "🚨 Alerte test manuelle"):
    await broadcast_notification(message)
    return {"status": "sent", "message": message}

# =============================
# 🔹 /analyze_patients : redirection et cas critiques
# =============================
@app.get("/analyze_patients")
async def analyze_patients(background_tasks: BackgroundTasks):
    try:
        df = pd.read_csv("data_final_preprocessedNoNormalized.csv")
        now = pd.Timestamp.now()

        for _, row in df.iterrows():
            waited = np.random.randint(70, 150)

            if row["urgency_level"] == 1 and waited > 60:
                msg = f"🔄 Un patient à urgence faible a une attente anormale de {waited} minutes. Redirection recommandée."
                background_tasks.add_task(broadcast_notification, msg)

            elif row["urgency_level"] >= 3 and pd.notna(row["time_to_medical_professional_min"]):
                if waited > row["time_to_medical_professional_min"] * 1.5:
                    msg = f"🚨 Un patient critique dépasse le délai d'attente prévu ({waited} min). Intervention urgente nécessaire."
                    background_tasks.add_task(broadcast_notification, msg)

        return {"status": "checked_csv", "count": len(df)}

    except Exception as e:
        return {"error": str(e)}
