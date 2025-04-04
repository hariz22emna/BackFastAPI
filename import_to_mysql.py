import pandas as pd
from sqlalchemy import create_engine

# === Étape 1 : Charger ton fichier CSV prétraité
df = pd.read_csv("C:/Users/ASUS/Desktop/pfa/FastAPI/data_final_preprocessedNoNormalized.csv")  

# === Étape 2 : Configuration de ta base MySQL
host = "localhost"
port = 3306
user = "root"
password = "" 
database = "pfadataset"    

# === Étape 3 : Connexion SQLAlchemy
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")

# === Étape 4 : Exporter les données dans une table
df.to_sql("donnees_pretraitees", con=engine, if_exists="replace", index=False)

print("Données importées dans MySQL avec succès !")
