import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import subprocess
import dagshub
import dvc.api

# 1. Connexion à DagsHub
repo_owner = "camm22"
repo_name = "MLOps_final_project"
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

def get_metadata():
    # Récupère le commit Git actuel
    commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    # Récupère la version des données via DVC
    data_url = dvc.api.get_url('data/iris.csv')
    return commit_hash, data_url

def train():
    mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
    
    with mlflow.start_run(run_name="Logistic Regression v1"):
        # Chargement des données
        df = pd.read_csv("data/iris.csv")
        X = df.drop("target", axis=1)
        y = df["target"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Modèle
        params = {"C": 1.0, "max_iter": 200}
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        # Métriques
        acc = accuracy_score(y_test, model.predict(X_test))
        
        # Logging des exigences
        commit, dvc_url = get_metadata()
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_param("git_commit", commit)
        mlflow.log_param("dvc_data_version", dvc_url)
        
        # Enregistrement dans le Registry
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="iris_logistic_model"
        )
        print(f"Entraînement réussi ! Accuracy: {acc}")

if __name__ == "__main__":
    train()
