# 1. Image de base Python
FROM python:3.12-slim

# 2. Définition du dossier de travail
WORKDIR /app

# 3. Copie du fichier de dépendances
# On remonte d'un cran pour prendre le requirements.txt à la racine
COPY requirements.txt .

# 4. Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copie du code du backend
COPY . .

# 6. Exposition du port utilisé par FastAPI
EXPOSE 8000

# 7. Commande pour lancer l'API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
