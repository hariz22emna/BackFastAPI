import pandas as pd
import requests

# 1. Lire les données
df = pd.read_csv("C:/Users/ASUS/Desktop/pfa/FastAPI/data_final_preprocessedNoNormalized.csv")

# 2. Vérifie les valeurs uniques (optionnel pour debug)
print("Valeurs possibles dans patient_outcome :", df["patient_outcome"].unique())

# 3. Encodage correct (très important !)
mapping_outcome = {
    "Discharged": 0,
    "Admitted": 1,
    "Left Without Being Seen": 2
}
df["patient_outcome"] = df["patient_outcome"].map(mapping_outcome)

# 4. Vérifie que le mapping a fonctionné (doit afficher 0, 1, 2, pas de string)
print("Extrait après encodage :")
print(df["patient_outcome"].head(10))

# 5. Sélection des colonnes dans le bon ordre
features_batch = df[[
    "urgency_level",
    "nurse-to-patient_ratio",
    "specialist_availability",
    "time_to_registration_min",
    "time_to_medical_professional_min",
    "available_beds_%"
]].values.tolist()


# 6. Envoyer à l’API
url = "http://127.0.0.1:8000/batch_predict"
response = requests.post(url, json={"features_batch": features_batch})

# 7. Résultat
print("✅ Résultat de l'API :")
if response.status_code == 200:
    print("✅ Prédictions :", response.json())
else:
    print("❌ Erreur API (status code):", response.status_code)
    print("↪️ Réponse brute :", response.text)
