import mlflow
import os

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/camm22/MLOps_final_project.mlflow"
MODEL_NAME = "iris_logistic_model"
ALIAS_TO_CHECK = "staging"  # On cherche l'alias au lieu du stage
ACCURACY_THRESHOLD = 0.90

def check_model_performance():
    client = mlflow.tracking.MlflowClient()
    
    try:
        # On récupère la version via son ALIAS
        model_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS_TO_CHECK)
        run_id = model_version.run_id
        
        metrics = client.get_run(run_id).data.metrics
        accuracy = metrics.get("accuracy", 0)
        
        print(f"Modèle version {model_version.version} - Accuracy: {accuracy}")
        
        if accuracy >= ACCURACY_THRESHOLD:
            print("✅ Quality Gate passée !")
            return True
        else:
            print(f"❌ Quality Gate échouée : Accuracy {accuracy} < {ACCURACY_THRESHOLD}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors de la récupération du modèle : {e}")
        return False

if __name__ == "__main__":
    if not check_model_performance():
        exit(1)