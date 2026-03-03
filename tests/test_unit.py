import pandas as pd
import os
import subprocess

def get_git_revision_hash():
    """Fonction utilitaire pour récupérer le hash Git."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "no-git-info"

def test_data_columns():
    """Vérifie que le dataset a les bonnes colonnes."""
    df = pd.read_csv("data/iris.csv")
    expected_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'target']
    assert all(col in df.columns for col in expected_columns)

def test_data_not_empty():
    """Vérifie que le dataset n'est pas vide."""
    df = pd.read_csv("data/iris.csv")
    assert len(df) > 0

def test_git_hash_format():
    """Vérifie que la récupération du commit Git fonctionne."""
    git_hash = get_git_revision_hash()
    assert isinstance(git_hash, str)
    assert len(git_hash) == 40 or git_hash != "no-git-info"
