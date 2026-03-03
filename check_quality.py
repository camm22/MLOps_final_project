import mlflow
import os

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/camm22/MLOps_final_project.mlflow"
MODEL_NAME = "iris_logistic_model"
ALIAS_TO_CHECK = "staging"
ACCURACY_THRESHOLD = 0.90

def check_model_performance():
    client = mlflow.tracking.MlflowClient()
    
    try:
        # On cherche la version qui porte l'alias 'staging'
        # Si l'alias n'est pas encore mis sur DagsHub, cette ligne lèvera une exception
        model_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS_TO_CHECK)
        run_id = model_version.run_id
        
        # Récupération des métriques du run associé
        run_data = client.get_run(run_id).data
        accuracy = run_data.metrics.get("accuracy", 0)
        
        print(f"Vérification du modèle Candidat (v{model_version.version})")
        print(f"Accuracy trouvée : {accuracy:.4f} (Seuil requis : {ACCURACY_THRESHOLD})")
        
        if accuracy >= ACCURACY_THRESHOLD:
            print("✅ Quality Gate passée avec succès !")
            return True
        else:
            print(f"❌ Échec : L'accuracy ({accuracy}) est trop faible.")
            return False
            
    except Exception as e:
        print(f"❌ Erreur : L'alias '{ALIAS_TO_CHECK}' n'a pas été trouvé sur DagsHub.")
        print("Vérifie que tu as bien ajouté l'alias 'staging' manuellement dans l'interface.")
        return False

if __name__ == "__main__":
    if not check_model_performance():
        exit(1)