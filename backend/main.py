from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
import pandas as pd
import os

app = FastAPI(title="MLOps Iris API")

# Configuration de l'accès MLflow
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/camm22/MLOps_final_project.mlflow"

MODEL_NAME = "iris_logistic_model"
MODEL_STAGE = os.getenv("MODEL_STAGE", "@staging")

model = None

# On essaie de charger le modèle au démarrage
try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}{MODEL_STAGE}")
    print(f"Modèle {MODEL_NAME} chargé avec succès depuis {MODEL_STAGE}")
except Exception as e:
    print(f"Attention : Impossible de charger le modèle ({e})")

@app.get("/")
def home():
    return {
        "message": "API Iris en ligne", 
        "model_status": "Ready" if model else "Model not loaded",
        "stage": MODEL_STAGE
    }

@app.post("/predict")
def predict(data: dict):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    
    try:
        df = pd.DataFrame([data["features"]])
        prediction = model.predict(df)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))