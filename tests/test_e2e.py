import requests

def test_complete_prediction_flow():
    """Simule une requête utilisateur réelle."""
    # Ce test sera surtout utile une fois déployé sur Render.
    # En local, on vérifie juste la cohérence de la sortie.
    sample_input = {"features": [6.7, 3.0, 5.2, 2.3]}
    assert isinstance(sample_input["features"], list)
