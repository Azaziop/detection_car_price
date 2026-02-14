FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de requirements
COPY requirements-docker.txt ./requirements.txt

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY . .

# Créer les répertoires nécessaires
RUN mkdir -p mlflow/mlruns models artifacts data/raw

# Exposer les ports
EXPOSE 8501 5000

# Script de démarrage par défaut
CMD ["streamlit", "run", "main_mlflow.py", "--server.port=8501", "--server.address=0.0.0.0"]
