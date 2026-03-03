import mlflow
import os

# Configuration
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/camm22/MLOps_final_project.mlflow"
MODEL_NAME = "iris_logistic_model"
ACCURACY_THRESHOLD = 0.90 # Notre Quality Gate

def check_model_performance():
    client = mlflow.tracking.MlflowClient()
    
    # On récupère la version du modèle en 'Staging'
    latest_version = client.get_latest_versions(MODEL_NAME, stages=["Staging"])[0]
    run_id = latest_version.run_id
    
    # On récupère l'accuracy enregistrée lors de l'entraînement
    metrics = client.get_run(run_id).data.metrics
    accuracy = metrics.get("accuracy", 0)
    
    print(f"Modèle version {latest_version.version} - Accuracy: {accuracy}")
    
    if accuracy >= ACCURACY_THRESHOLD:
        print("✅ Quality Gate passée !")
        return True
    else:
        print("❌ Quality Gate échouée : Accuracy trop faible.")
        return False

if __name__ == "__main__":
    if not check_model_performance():
        exit(1) # Arrête le pipeline en cas d'échec
