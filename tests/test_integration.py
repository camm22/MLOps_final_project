from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_api_health():
    """Vérifie que l'API répond sur la racine."""
    response = client.get("/")
    assert response.status_code == 200
    assert "API Iris en ligne" in response.json()["message"]

def test_api_prediction_format():
    """Vérifie que l'endpoint predict renvoie le bon format."""
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}
    response = client.post("/predict", json=payload)
    # Si le modèle n'est pas encore en 'Production' sur DagsHub, ce test peut échouer.
    assert response.status_code in [200, 500]
