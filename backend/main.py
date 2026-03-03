from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
import os

app = FastAPI(title="MLOps Iris API")

# Configuration de l'accès MLflow
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/camm22/MLOps_final_project.mlflow"

# On charge le modèle qui a le tag 'Production'
model_name = "iris_logistic_model"
model_version = "Production"
model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")

@app.get("/")
def home():
    return {"message": "API Iris en ligne", "model_stage": model_version}

@app.post("/predict")
def predict(data: dict):
    # 'data' doit être un dictionnaire avec les 4 caractéristiques d'Iris
    # Exemple: {"features": [5.1, 3.5, 1.4, 0.2]}
    df = pd.DataFrame([data["features"]])
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}
