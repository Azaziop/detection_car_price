# 🚗 Documentation Complète du Projet Car Price Prediction MLOps Pipeline

## 📋 Table des Matières

1. [Vue d'Ensemble](#-vue-densemble)
2. [Architecture du Système](#️-architecture-du-système)
3. [Technologies Utilisées](#️-technologies-utilisées)
4. [Structure du Projet](#-structure-du-projet)
5. [Configuration et Installation](#️-configuration-et-installation)
6. [Pipeline de Données](#-pipeline-de-données)
7. [Modèle de Machine Learning](#-modèle-de-machine-learning)
8. [Orchestration avec Airflow](#-orchestration-avec-airflow)
9. [Tracking avec MLflow](#-tracking-avec-mlflow)
10. [Services et Ports](#-services-et-ports)
11. [Utilisation](#-utilisation)
12. [Monitoring et Reporting](#-monitoring-et-reporting)
13. [Troubleshooting](#-troubleshooting)
14. [Déploiement en Production](#-déploiement-en-production)

---

## 🎯 Vue d'Ensemble

Ce projet implémente un **pipeline MLOps complet** pour la prédiction des prix des voitures au Maroc. Il automatise l'ensemble du cycle de vie du machine learning, de la collecte des données jusqu'au déploiement en production.

### Objectifs Principaux

- ✅ **Automatisation complète** du pipeline ML (entraînement, évaluation, déploiement)
- ✅ **Traçabilité** de tous les modèles et expériences avec MLflow
- ✅ **Orchestration robuste** des tâches avec Apache Airflow
- ✅ **Containerisation** pour la reproductibilité avec Docker
- ✅ **Promotion automatique** des modèles (None → Staging → Production)
- ✅ **Monitoring** et reporting automatisé
- ✅ **Interface Web** interactive avec Streamlit
- ✅ **Tests unitaires** et intégration continue

### Fonctionnalités Clés

---

## 🏗️ Architecture du Système

### Infrastructure Docker

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Compose Stack                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Airflow    │  │   MLflow     │  │  PostgreSQL  │      │
│  │  Webserver   │  │   Server     │  │   (Airflow)  │      │
│  │  :8080       │  │   :5050      │  │   :54322     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Airflow    │  │   Airflow    │  │  PostgreSQL  │      │
│  │  Scheduler   │  │  Triggerer   │  │   (MLflow)   │      │
│  │              │  │              │  │   :54323     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Airflow    │  │   Airflow    │  │   Streamlit  │      │
│  │   Worker     │  │     Init     │  │     App      │      │
│  │  (Celery)    │  │              │  │   :8501      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌──────────────────────────────────────────────────┐       │
│  │              Redis (Message Broker)               │       │
│  │                  :6379                            │       │
│  └──────────────────────────────────────────────────┘       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Flux de Données du Pipeline ML

```
Raw Data (CSV 24,776 lignes)
    ↓
┌─────────────────────────────┐
│ 1. check_data_quality       │ ← Validation (4.6% missing)
---

## 🛠️ Technologies Utilisées

### Orchestration & Workflow
- **Apache Airflow 2.9.3** - Orchestration du pipeline ML
- **Celery** - Exécution distribuée des tâches
- **Redis 7.2** - Message broker pour Celery
---

## 📁 Structure du Projet

```
PythonProject9/
├── README.md                           # Ce fichier - Documentation complète
├── requirements.txt                    # Dépendances Python
├── requirements-airflow.txt            # Dépendances Airflow + ML
├── params.yaml                         # Hyperparamètres du modèle
├── docker-compose-full.yml             # Stack Docker complète
├── Dockerfile.airflow                  # Image custom Airflow + ML
├── docker-start-full.sh                # Script démarrage stack
├── docker-stop-full.sh                 # Script arrêt stack
├── docker-reset-full.sh                # Script reset complet
│
├── airflow/
│   ├── dags/
│   │   └── car_price_ml_pipeline.py    # DAG principal (7 tâches)
│   ├── logs/                           # Logs Airflow
│   │   └── dag_id=car_price_predictor_pipeline/
│   └── config/
│       └── airflow.cfg                 # Configuration Airflow
│
├── scripts/
│   ├── train_with_mlflow.py            # Pipeline ML complet
│   │                                   # (CarPricePipeline class)
│   └── load_model_mlflow.py            # Chargement modèles
│
├── data/
│   └── raw/
│       └── avito_car_dataset_ALL.csv   # Dataset source (24,776 lignes)
│
├── models/
│   ├── car_model.pkl                   # Modèle RandomForest entraîné
│   ├── scaler.pkl                      # StandardScaler pour features
│   └── encoders.pkl                    # LabelEncoders pour catégorielles
│
├── artifacts/
│   ├── feature_importance.png          # Graphique importance features
│   ├── feature_importance.csv          # Données importance
│   ├── predictions_plot.png            # Prédictions vs réel
│   ├── residuals_plot.png              # Analyse des résidus
│   ├── feature_info.json               # Métadonnées features
│   └── price_scaler_info.json          # Info scaler prix
│
├── reports/
│   └── pipeline_report_*.json          # Rapports d'exécution pipeline
│
├── mlflow/
│   └── mlruns/                         # Artifacts MLflow locaux
│       └── <experiment_id>/
│           └── <run_id>/
│               ├── artifacts/
│               ├── metrics/
│               ├── params/
│               └── tags/
│
├── tests/
│   ├── test_pipeline.py                # Tests unitaires pipeline
│   ├── test_integration.py             # Tests d'intégration
│   └── test_car_pipeline.py            # Tests CarPricePipeline
│
└── main_mlflow.py                      # Application Streamlit
```

---

## ⚙️ Configuration et Installation

### Méthode 1: Installation Docker (Recommandée - Production Ready)

#### 1. Cloner le Repository

```bash
git clone https://github.com/Azaziop/detection_car_price.git
cd detection_car_price
```

#### 2. Vérifier Docker

```bash
# Vérifier que Docker est installé et en cours d'exécution
docker --version
docker-compose --version

# Démarrer Docker Desktop si nécessaire
```

#### 3. Démarrer la Stack Complète

```bash
# Donner les permissions d'exécution
chmod +x docker-start-full.sh docker-stop-full.sh docker-reset-full.sh

# Démarrer tous les services (MLflow + Streamlit + Airflow)
./docker-start-full.sh

# Attendre 2-3 minutes que tous les services démarrent
# Suivre les logs en temps réel
docker-compose -f docker-compose-full.yml logs -f
```

#### 4. Vérifier les Services

```bash
# Status de tous les services
docker-compose -f docker-compose-full.yml ps

# Les 9 services devraient être "healthy" ou "running"
```

#### 5. Accéder aux Interfaces

- **Airflow**: http://localhost:8080 (admin / airflow)
- **MLflow**: http://localhost:5050
- **Streamlit**: http://localhost:8501

### Méthode 2: Installation Locale (Développement)

#### 1. Créer un Environnement Virtuel

**Avec Python 3.11 (recommandé pour Airflow):**
```bash
# Option 1: Si Python 3.11 est installé via Homebrew
python3.11 -m venv .venv

# Option 2: Si pyenv est utilisé
pyenv install 3.11.7
pyenv local 3.11.7
python -m venv .venv

# Activer l'environnement
source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate
```

#### 2. Installer les Dépendances

```bash
# Dépendances de base
pip install -r requirements.txt

# Ou pour Airflow + ML
pip install -r requirements-airflow.txt

# Développement (optionnel)
pip install -r requirements/requirements-dev.txt
```

#### 3. Configuration Airflow (Optionnel)

```bash
# Définir le répertoire Airflow
export AIRFLOW_HOME=$(pwd)/airflow

# Initialiser la base de données
airflow db init

# Créer un utilisateur admin
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Démarrer Airflow
airflow standalone+ Feature Engineering
│    - Preprocess: 7,524 rows │    (27 features: 9 cat + 3 num + 15 bin)
│    - Train: RandomForest    │
│    - MLflow: Log metrics    │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│ 3. evaluate_model           │ ← Fetch metrics from MLflow
│    - R² = 0.73              │   Quality check: R²≥0.70 ✅
│    - RMSE = 0.52            │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│ 4. promote_to_staging       │ ← Transition to "Staging"
│    - Find version by run_id │   MLflow Model Registry
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│ 5. validate_staging_model   │ ← Validate model metadata
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│ 6. promote_to_production    │ ← Transition to "Production"
│    - Archive old version    │   Deploy new model
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│ 7. send_pipeline_report     │ ← Generate JSON report
│    - Aggregate all metrics  │   Save to reports/
└─────────────────────────────┘
```

### Communication entre Services

```
Airflow Tasks ←→ MLflow Server (http://172.18.0.4:5000)
      ↓              ↓
   XCom Data    PostgreSQL (tracking)
      ↓              ↓
File System ←→ Artifacts Storage
(models, artifacts, reports)price_scaler_info.json │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Interface Streamlit        │
│  (main.py / main_mlflow.py) │
└─────────────────────────────┘
         │
         ▼
    Prédictions
```

## 📦 Prérequis

- **Python 3.11** (recommandé) ou Python 3.8-3.11
  - ⚠️ **Python 3.12 non compatible** avec Apache Airflow sur macOS
  - Voir [PYTHON_3.11_MIGRATION.md](PYTHON_3.11_MIGRATION.md) pour migrer depuis 3.12
- pip ou conda
- Git

## 🚀 Installation

### 1. Cloner le repository

```bash
git clone https://github.com/Azaziop/detection_car_price.git
cd detection_car_price
```

### 2. Créer un environnement virtuel

**Avec Python 3.11 (recommandé pour Airflow):**
```bash
# Option 1: Si Python 3.11 est installé via Homebrew
python3.11 -m venv .venv

# Option 2: Si pyenv est utilisé
pyenv install 3.11.7
pyenv local 3.11.7
python -m venv .venv

# Activer l'environnement
source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate
```

**Avec Python 3.12 (limitations):**
```bash
python -m venv .venv
source .venv/bin/activate
# ⚠️ Note: Airflow ne fonctionnera pas - voir PYTHON_3.11_MIGRATION.md
```

### 3. Installer les dépendances

```bash
pip install -r requirements/requirements.txt
```

### 4. (Optionnel) Installation pour développement

```bash
pip install -r requirements/requirements-dev.txt
```

---

## 🔄 Pipeline de Données

### Vue d'Ensemble du Pipeline

Le pipeline `CarPricePipeline` (dans `scripts/train_with_mlflow.py`) exécute 6 étapes principales:

1. **Chargement** → 2. **Prétraitement** → 3. **Feature Engineering** → 4. **Entraînement** → 5. **Évaluation** → 6. **Sauvegarde**

### 1. Chargement des Données (`load_data`)

```python
def load_data(self, filepath='/opt/airflow/project/data/raw/avito_car_dataset_ALL.csv'):
    df = pd.read_csv(filepath, encoding='latin1')
    # Input: 24,776 lignes × 32 colonnes
    return df
```

**Caractéristiques**:
- Source: Dataset Avito Maroc
- Format: CSV avec encodage latin1
- Colonnes: 32 (caractéristiques véhicules + prix)

### 2. Prétraitement (`preprocess_data`)

#### a) Gestion des Valeurs Manquantes

```python
# Catégorielles - remplissage par mode (valeur la plus fréquente)
df['Origine'] = df['Origine'].fillna(df['Origine'].mode()[0])
df['Première main'] = df['Première main'].fillna(df['Première main'].mode()[0])
df['État'] = df['État'].fillna(df['État'].mode()[0])

# Numériques - remplissage par médiane
df['Nombre de portes'] = df['Nombre de portes'].fillna(df['Nombre de portes'].median())
df['Puissance fiscale'] = df['Puissance fiscale'].fillna(df['Puissance fiscale'].median())

# Features binaires - remplissage par 0 (non équipé)
binary_features = ['Jantes aluminium', 'Airbags', 'Climatisation', ...]
for col in binary_features:
    df[col] = df[col].fillna(0)
```

#### b) Suppression des Doublons

```python
df = df.drop_duplicates()
# Réduction: 24,776 → ~24,500 lignes
```

#### c) Détection et Suppression des Outliers (IQR Method)

```python
# Méthode Interquartile Range
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtrage des outliers
for col in numeric_cols:
    df = df[(df[col] >= lower_bound[col]) & (df[col] <= upper_bound[col])]

# Résultat final: ~7,524 lignes (données propres)
```

#### d) Suppression de Colonnes Inutiles

```python
# Colonnes à supprimer
drop_cols = ['Unnamed: 0', 'Lien', 'Secteur']
df = df.drop(columns=drop_cols, errors='ignore')
```

### 3. Feature Engineering (`prepare_features`)

#### Features Numériques (3)
- **Année-Modèle**: Age du véhicule (transformé en années depuis fabrication)
- **Kilométrage**: Distance parcourue
- **Puissance fiscale**: Puissance du moteur

```python
numeric_features = ['Année-Modèle', 'Kilométrage', 'Puissance fiscale']
```

#### Features Catégorielles (9) - Label Encoding
```python
categorical_features = [
    'Marque',              # Constructeur (Dacia, Renault, Peugeot, ...)
    'Modèle',              # Modèle du véhicule
    'Type de carburant',   # Essence, Diesel, Hybride, Électrique
    'Boite de vitesses',   # Manuelle, Automatique
    'Origine',             # WW au Maroc, Importée
    'Première main',       # Oui, Non
    'État',                # Excellent, Très bon, Bon, Correct
    'Ville',               # Localisation géographique
    'Nombre de portes'     # 2, 3, 4, 5 portes
]

# Encodage avec LabelEncoder
from sklearn.preprocessing import LabelEncoder
encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Sauvegarde des encoders
joblib.dump(encoders, 'models/encoders.pkl')
```

#### Features Binaires (15)
```python
binary_features = [
    'Jantes aluminium', 'Airbags', 'Climatisation',
    'Système de navigation/GPS', 'Toit ouvrant', 'Sièges cuir',
    'Radar de recul', 'Caméra de recul', 'Vitres électriques',
    'ABS', 'ESP', 'Régulateur de vitesse', 'Limiteur de vitesse',
    'CD/MP3/Bluetooth', 'Ordinateur de bord',
    'Verrouillage centralisé à distance'
]
# Valeurs: 0 (non équipé) ou 1 (équipé)
```

**Total: 27 features** (3 numériques + 9 catégorielles + 15 binaires)

#### Normalisation (StandardScaler)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Formule: z = (x - μ) / σ
# μ = moyenne, σ = écart-type

# Sauvegarde du scaler
joblib.dump(scaler, 'models/scaler.pkl')
```

### 4. Entraînement du Modèle (`train_model`)

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Split train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled,
    test_size=0.2,
    random_state=42
)

# Configuration du modèle (depuis params.yaml)
model = RandomForestRegressor(
    n_estimators=100,        # 100 arbres de décision
    max_depth=20,            # Profondeur max: 20 niveaux
    min_samples_split=5,     # Min 5 échantillons pour split
    min_samples_leaf=2,      # Min 2 échantillons par feuille
    max_features='sqrt',     # √27 ≈ 5 features par split
    random_state=42,         # Reproductibilité
    n_jobs=-1                # Utiliser tous les CPU
)

# Entraînement
model.fit(X_train, y_train)

# Résultats:
# - Train set: 6,019 samples
# - Test set: 1,505 samples
```

### 5. Évaluation (`evaluate`)

```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Prédictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Métriques d'entraînement
train_r2 = r2_score(y_train, y_train_pred)      # 0.8689 (86.89%)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

# Métriques de test
test_r2 = r2_score(y_test, y_test_pred)         # 0.7299 (72.99%)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))  # 0.5188
test_mae = mean_absolute_error(y_test, y_test_pred)           # 0.3872
```

**Interprétation**:
- **R² Test = 0.73**: Le modèle explique 73% de la variance des prix
- **RMSE = 0.52**: Erreur quadratique moyenne (sur prix normalisés)
- **MAE = 0.39**: Erreur absolue moyenne (sur prix normalisés)
- **Légère sur-apprentissage**: R² train (0.87) > R² test (0.73), mais acceptable

### 6. Sauvegarde et Logging MLflow

```python
import mlflow
import mlflow.sklearn

with mlflow.start_run():
    # Log des paramètres
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 20)
    mlflow.log_param("min_samples_split", 5)
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))
    
    # Log des métriques
    mlflow.log_metric("train_r2", train_r2)
    mlflow.log_metric("test_r2", test_r2)
    mlflow.log_metric("test_rmse", test_rmse)
    mlflow.log_metric("test_mae", test_mae)
    
    # Log des artifacts
    mlflow.log_artifact("artifacts/feature_importance.png")
    mlflow.log_artifact("artifacts/predictions_plot.png")
    mlflow.log_artifact("artifacts/residuals_plot.png")
    
    # Enregistrement du modèle dans MLflow Registry
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="CarPricePredictor"
    )
    
    run_id = mlflow.active_run().info.run_id
```

### Artifacts Générés

```
artifacts/
├── feature_importance.png      # Barplot des 15 features les plus importantes
├── feature_importance.csv      # Données brutes importance
├── predictions_plot.png        # Scatter: Prédictions vs Valeurs réelles
├── residuals_plot.png          # Distribution des résidus (erreurs)
├── feature_info.json           # Métadonnées: noms, types, encodings
└── price_scaler_info.json      # Paramètres du scaler de prix (μ, σ)

models/
├── car_model.pkl               # Modèle RandomForest entraîné
├── scaler.pkl                  # StandardScaler pour features
└── encoders.pkl                # Dict de LabelEncoders
```

### Option 1: Lancer l'application Streamlit (Recommandé)

```bash
streamlit run main_mlflow.py
```

L'application s'ouvrira à `http://localhost:8501`

**Fonctionnalités de l'app:**
- 🎯 Formulaire pour entrer les caractéristiques du véhicule
- 💰 Prédiction du prix en DH marocain
- 📊 Visualisations des features importance
- 📈 Historique des prédictions

### Option 2: Utiliser le modèle en Python

```python
import joblib
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder

# Charger les artifacts
model = joblib.load('models/car_model.pkl')
scaler = joblib.load('models/scaler.pkl')

with open('artifacts/feature_info.json', 'r') as f:
    feature_info = json.load(f)

with open('artifacts/price_scaler_info.json', 'r') as f:
    price_scaler_info = json.load(f)

# Créer les encodeurs et préparer les données
# [Voir CODE_EXAMPLES.md pour l'exemple complet]

# Faire une prédiction
prediction = model.predict(X_scaled)
```

### Option 3: Réentraîner le modèle

#### Avec DVC:
```bash
dvc repro -f dvc/dvc.yaml
```

#### Ou directement:
```bash
python scripts/train_with_mlflow.py
```

### Option 4: Lancer les tests

```bash
pytest tests/ -v
pytest tests/ --cov=.  # Avec coverage
```

## 🔄 Pipeline de données

### Étapes du pipeline:

1. **Chargement** (`load_data`)
   - Lecture du CSV Avito Maroc
   - Encodage: latin1

2. **Nettoyage** (`preprocess_data`)
   - Imputation des valeurs manquantes
   - Suppression des doublons
   - Suppression des colonnes corrélées

3. **Encodage** (`encode_features`)
   - Label encoding pour variables catégoriques
   - OneHot encoding optionnel

4. **Normalisation** (`scale_features`)
   - StandardScaler pour features numériques

5. **Entraînement** (`train_model`)
   - Random Forest Regressor
   - Hyperparamètres optimisés

6. **Évaluation** (`evaluate`)
   - MAE, MSE, R² Score
   - Sauvegarde avec MLflow

### Configuration du pipeline

Voir `params.yaml`:
```yaml
train:
  test_size: 0.2
  random_state: 42
model:
  n_estimators: 100
  max_depth: 20
  min_samples_split: 5
  min_samples_leaf: 2
  max_features: 'sqrt'
```

## 📊 Résultats du modèle

Le modèle Random Forest entraîné achieves:
- **R² Score**: ~0.87
- **MAE (Mean Absolute Error)**: Environ 15-20% du prix moyen
- **Données**: 10,000+ véhicules Avito Maroc

### Features importantes:
1. Kilométrage
2. Année-Modèle
3. Marque du véhicule
4. État général
5. Puissance fiscale

## 📁 Structure du projet

```
detection_car_price/
├── README.md                      # Ce fichier
├── requirements/requirements.txt               # Dépendances pip
├── requirements/requirements-dev.txt           # Dépendances développement
├── params.yaml                    # Hyperparamètres du modèle
├── dvc/dvc.yaml                               # Pipeline DVC
├── pytest.ini                     # Configuration pytest
│
├── data/raw/avito_car_dataset_ALL.csv      # Dataset source
├── main.py                        # App Streamlit basique
├── main_mlflow.py                 # App Streamlit avec MLflow
├── scripts/train_with_mlflow.py   # Pipeline d'entraînement
├── finalpreoject.py               # Analyse EDA
├── scripts/load_model_mlflow.py   # Chargement des modèles
│
├── tests/                         # Suite de tests
│   ├── __init__.py
│   ├── test_pipeline.py
│   ├── test_integration.py
│   └── test_car_pipeline.py
│
├── mlflow/mlruns/                 # Artifacts MLflow
│   ├── 1/                         # Experiment 1
│   ├── 710723541858247182/        # Experiment 2
│   └── models/                    # Registered Models
│
├── reports/htmlcov/               # Coverage reports
└── __pycache__/                   # Cache Python
```

## 🛠️ Technologies utilisées

### Data & ML:
- **pandas** - Manipulation de données
- **NumPy** - Calculs numériques
- **scikit-learn** - Machine Learning
- **joblib** - Sérialisation de modèles

### MLOps:
- **MLflow** - Tracking d'expériences et versioning
- **DVC** - Gestion de données et pipelines

### Frontend:
- **Streamlit** - Interface web interactive

### Visualisation:
- **matplotlib** - Graphiques
- **seaborn** - Visualisations avancées
- **ydata-profiling** - Rapports EDA

### DevOps & Tests:
- **pytest** - Framework de test
- **PyYAML** - Gestion de fichiers YAML
- **skops** - Sérialisation scikit-learn

## 📈 Métriques MLflow

Les expériences sont trackées dans MLflow. Pour visualiser le dashboard:

```bash
mlflow ui
```

Puis accédez à `http://localhost:5000`

Vous verrez:
- Historique des entraînements
- Comparaison des métriques
- Paramètres utilisés
- Artifacts (modèles, scalers)

## 🧪 Tests

```bash
# Lancer tous les tests
pytest tests/ -v

# Tests avec coverage
pytest tests/ --cov=. --cov-report=html

# Tests spécifiques
pytest tests/test_pipeline.py -v
pytest tests/test_integration.py -v
```

## 📚 Documentation supplémentaire

- Voir [CODE_EXAMPLES.md](CODE_EXAMPLES.md) pour des exemples d'utilisation détaillés
- Rapport de profiling: [reports/profiling_rep.html](reports/profiling_rep.html)
- Coverage report: [reports/htmlcov/index.html](reports/htmlcov/index.html)

## 🔍 Analyse EDA

Un rapport complet de l'analyse exploratoire est généré dans `reports/profiling_rep.html`:

```bash
# Régénérer le rapport (optionnel)
python finalpreoject.py
```

Contient:
- Statistiques descriptives
- Distribution des variables
- Corrélations entre features
- Détection d'anomalies
- Valeurs manquantes

## 🐛 Troubleshooting

### Airflow crashe avec erreurs SIGSEGV

**Diagnostic**: Ce bug affecte **macOS ARM64 uniquement** (M1/M2/M3), toutes versions Python.

**Root Cause**: Gunicorn + macOS ARM64 incompatibilité (problème connu upstream).

**Solution définitive**: Utiliser Docker (voir section Airflow ci-dessus).

**Tentatives infructueuses documentées:**
- ❌ Migration Python 3.12 → 3.11 : N'a pas résolu le problème
- ❌ Configuration webserver_config.py avec workers sync : Échec
- ❌ Variables d'environnement GUNICORN_CMD_ARGS : Sans effet

**Verdict**: Airflow n'est actuellement **pas supporté nativement** sur macOS ARM64.

**Solution**: Migrer vers Python 3.11
```bash
# Voir le guide complet
cat PYTHON_3.11_MIGRATION.md

# Migration express
brew install python@3.11
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Alternatives**:
- Docker: `docker run -p 8080:8080 apache/airflow:2.8.1 standalone`
- Déploiement Linux/Cloud

Documentation complète: [PYTHON_3.11_MIGRATION.md](PYTHON_3.11_MIGRATION.md)

### L'app Streamlit ne démarre pas

```bash
# Vérifier les dépendances
pip install -r requirements/requirements.txt

# Réinstaller en cas de problème
pip install --force-reinstall -r requirements/requirements.txt
```

### Modèle non trouvé

Assurez-vous d'avoir entraîné le modèle:
```bash
python scripts/train_with_mlflow.py
# ou
dvc repro -f dvc/dvc.yaml
```

### Erreurs d'encodage CSV

Le dataset utilise l'encodage `latin1`. Ne le changez pas.

### Vérifier la version Python

```bash
# Dans le projet
python --version  # Doit afficher 3.11.x pour Airflow

# Changer de version avec pyenv
pyenv versions  # Lister les versions disponibles
pyenv local 3.11.7  # Utiliser 3.11 pour ce projet
```

## 🤝 Contribution

Les contributions sont bienvenues! Pour contribuer:

1. Fork le repository
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📝 License

Ce projet est open source et disponible sous la licence MIT.

## ⚠️ Prérequis Airflow

**Apache Airflow nécessite Python 3.11 sur macOS.**

Si vous utilisez Python 3.12, suivez le guide de migration:
```bash
# Guide complet de migration
cat PYTHON_3.11_MIGRATION.md

# Migration rapide (15 min)
brew install python@3.11
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Voir: [PYTHON_3.11_MIGRATION.md](PYTHON_3.11_MIGRATION.md)

### 🐳 Stack MLOps Complète avec Docker

**Solution professionnelle : Tous les services dans Docker**

Cette stack intègre MLflow, Streamlit, Airflow et PostgreSQL dans des containers isolés.

**Démarrage rapide:**
```bash
# Démarrer toute la stack (MLflow + Streamlit + Airflow)
./docker-start-full.sh

# Arrêter la stack
./docker-stop-full.sh

# Réinitialiser complètement (si problèmes)
./docker-reset-full.sh
```

**Services disponibles:**
- **MLflow**: http://localhost:5000 - Tracking d'expériences
- **Streamlit**: http://localhost:8501 - Interface de prédiction
- **Airflow**: http://localhost:8080 - Orchestration (admin / admin)

**Avantages Docker:**
- ✅ Isolation complète des services
- ✅ Reproductibilité garantie
- ✅ Pas de conflits de dépendances
- ✅ Production-ready
- ✅ Fonctionne sur macOS ARM64 (M1/M2/M3)

**Commandes utiles:**
```bash
# Voir les logs en temps réel
docker compose -f docker-compose-full.yml logs -f

# Logs d'un service spécifique
docker compose -f docker-compose-full.yml logs -f streamlit
docker compose -f docker-compose-full.yml logs -f mlflow
docker compose -f docker-compose-full.yml logs -f airflow-webserver

# Statut des services
docker compose -f docker-compose-full.yml ps

# Redémarrer un service
docker compose -f docker-compose-full.yml restart streamlit
```

**Interface Web**: http://localhost:8080 (consulter le terminal pour les identifiants)

**Alternative - Mode séparé:**
```bash
# Terminal 1 - Scheduler
export AIRFLOW_HOME=$(pwd)/airflow
airflow scheduler

# Tester sans interface web
airflow dags test car_price_predictor_pipeline $(date +%Y-%m-%d)
```

### Pipeline Automatisé

Le DAG `car_price_predictor_pipeline` exécute automatiquement:

1. ✅ **Vérification données** - Qualité et volume
2. 🚀 **Entraînement** - Model training avec MLflow
3. 📊 **Évaluation** - Métriques R² et RMSE
4. 📦 **Staging** - Promotion si critères OK (R²>0.80, RMSE<50k)
5. 🧪 **Validation** - Tests en Staging
6. 🎯 **Production** - Déploiement automatique
7. 📧 **Rapport** - Génération rapport JSON

**Planification**: Hebdomadaire (modifiable dans le DAG)

Voir la documentation complète: [airflow/README_AIRFLOW.md](airflow/README_AIRFLOW.md)

---

## 🎯 Objectifs futurs

- [x] Pipeline MLOps avec Airflow
- [ ] Déploiement sur cloud (AWS/GCP/Azure)
- [ ] API REST avec FastAPI
- [ ] Dashboard de monitoring
- [ ] A/B testing de modèles
- [ ] Prédictions batch
- [ ] Explainability avec SHAP

---

**Dernière mise à jour**: Février 2026  
**Version**: 1.1.0 (avec Airflow)
