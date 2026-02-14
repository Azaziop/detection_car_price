# 🚗 Documentation Complète - Car Price Prediction MLOps Pipeline

> Documentation technique exhaustive du projet MLOps de prédiction des prix de voitures

**Dernière mise à jour**: 14 février 2026  
**Version**: 2.0.0

---

## 📚 Documents

Ce projet contient plusieurs documents de documentation:

1. **[README.md](README.md)** - Documentation principale et guide de démarrage rapide
2. **Ce fichier (DOCUMENTATION_COMPLETE.md)** - Documentation technique complète et détaillée
3. **[DOCKER_GUIDE.md](DOCKER_GUIDE.md)** - Guide spécifique Docker
4. **[CODE_EXAMPLES.md](CODE_EXAMPLES.md)** - Exemples de code
5. **[PYTHON_3.11_MIGRATION.md](PYTHON_3.11_MIGRATION.md)** - Guide de migration Python

---

## 🤖 Modèle de Machine Learning

### Algorithme: Random Forest Regressor

**Justification du choix**:

1. **Robustesse**: Résistant au surapprentissage grâce à l'ensemble d'arbres
2. **Performance**: Excellentes performances sur données tabulaires
3. **Interprétabilité**: Feature importance facilement extractible
4. **Non-linéarité**: Capture les relations complexes entre variables
5. **Pas de normalisation obligatoire**: Fonctionne bien sans preprocessing poussé

### Configuration du Modèle

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,         # Nombre d'arbres dans la forêt
    max_depth=20,             # Profondeur maximale de chaque arbre
    min_samples_split=5,      # Échantillons minimum pour split un nœud
    min_samples_leaf=2,       # Échantillons minimum par feuille
    max_features='sqrt',      # √features pour chaque split
    random_state=42,          # Graine pour reproductibilité
    n_jobs=-1,                # Utiliser tous les CPU disponibles
    verbose=0                 # Pas de logs pendant l'entraînement
)
```

**Paramètres expliqués**:

- **n_estimators=100**: Plus d'arbres = meilleure performance, mais temps d'entraînement ↑
- **max_depth=20**: Limite la profondeur pour éviter le surapprentissage
- **min_samples_split=5**: Empêche les splits sur peu de données
- **min_samples_leaf=2**: Assure des feuilles avec au moins 2 échantillons
- **max_features='sqrt'**: √27 ≈ 5 features aléatoires par split (diversité)

### Métriques de Performance

#### R² Score (Coefficient de Détermination)

```python
r2_score = r2_score(y_test, y_pred)
```

**Formule**: R² = 1 - (SS_res / SS_tot)
- SS_res = Somme des carrés des résidus
- SS_tot = Somme des carrés totaux

**Interprétation**:
- R² = 1.0 → Modèle parfait (prédit exactement)
- R² = 0.7 → Modèle explique 70% de la variance
- R² = 0.0 → Modèle pas mieux qu'une moyenne
- R² < 0.0 → Modèle pire qu'une moyenne

**Notre résultat**: R² = 0.73 ✅
- 73% de la variance des prix est expliquée par le modèle
- Performance satisfaisante pour un problème de prédiction de prix

#### RMSE (Root Mean Squared Error)

```python
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
```

**Formule**: RMSE = √(Σ(y_i - ŷ_i)² / n)

**Caractéristiques**:
- Unité: Même que la variable cible (prix normalisé)
- Pénalise fortement les grandes erreurs (au carré)
- Sensible aux outliers

**Notre résultat**: RMSE = 0.52
- Sur des prix normalisés (μ=0, σ=1)
- Erreur typique de 0.52 écart-type

#### MAE (Mean Absolute Error)

```python
mae = mean_absolute_error(y_test, y_pred)
```

**Formule**: MAE = Σ|y_i - ŷ_i| / n

**Caractéristiques**:
- Interprétation directe: erreur moyenne
- Moins sensible aux outliers que RMSE
- Plus robuste

**Notre résultat**: MAE = 0.39
- Erreur absolue moyenne de 0.39 écart-type
- Plus petit que RMSE → Pas d'outliers majeurs

### Feature Importance

```python
# Extraction des importances
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
```

**Top 10 Features** (% d'importance):

| Rang | Feature | Importance | Catégorie |
|------|---------|------------|-----------|
| 1 | Année-Modèle | 34.29% | Numérique |
| 2 | Boite de vitesses | 10.07% | Catégorielle |
| 3 | Marque | 9.91% | Catégorielle |
| 4 | Modèle | 9.61% | Catégorielle |
| 5 | Puissance fiscale | 9.35% | Numérique |
| 6 | Kilométrage | 6.77% | Numérique |
| 7 | Ville | 3.80% | Catégorielle |
| 8 | Première main | 3.55% | Catégorielle |
| 9 | Type de carburant | 3.45% | Catégorielle |
| 10 | ESP | 2.53% | Binaire |

**Insights**:
- **Année-Modèle** est de loin la feature la plus importante (34%)
- Les **3 features numériques** totalisent ~50% de l'importance
- La **transmission** (manuelle/auto) a un impact significatif (10%)
- Les **équipements de sécurité** (ESP, ABS) ont un poids modéré

### Visualisations Générées

#### 1. Feature Importance Plot
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
sns.barplot(data=top_features, y='feature', x='importance')
plt.title('Top 15 Features - Importance dans le Modèle')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('artifacts/feature_importance.png', dpi=300, bbox_inches='tight')
```

#### 2. Predictions vs Actual
```python
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Prix Réels (normalisés)')
plt.ylabel('Prix Prédits (normalisés)')
plt.title('Prédictions vs Valeurs Réelles')
plt.legend()
plt.savefig('artifacts/predictions_plot.png', dpi=300, bbox_inches='tight')
```

**Analyse**:
- Points proches de la diagonale = bonnes prédictions
- Dispersion = incertitude du modèle
- Points éloignés = erreurs de prédiction

#### 3. Residuals Plot
```python
residuals = y_test - y_pred

plt.figure(figsize=(12, 5))

# Distribution des résidus
plt.subplot(1, 2, 1)
plt.hist(residuals, bins=50, edgecolor='black')
plt.xlabel('Résidus')
plt.ylabel('Fréquence')
plt.title('Distribution des Résidus')

# Q-Q Plot
plt.subplot(1, 2, 2)
from scipy import stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot (Normalité des Résidus)')

plt.tight_layout()
plt.savefig('artifacts/residuals_plot.png', dpi=300, bbox_inches='tight')
```

**Analyse**:
- Distribution proche d'une Normale → Bon signe
- Résidus centrés autour de 0 → Pas de biais
- Outliers visibles → Prédictions difficiles sur certaines voitures

### Amélioration Possible du Modèle

**Pistes d'optimisation**:

1. **Hyperparameter Tuning**:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [15, 20, 25],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
```

2. **Feature Engineering Avancé**:
- Interaction features (Marque × Modèle)
- Age du véhicule² (non-linéarité)
- Kilométrage par année
- Ratio puissance/poids

3. **Algorithmes Alternatifs**:
- **XGBoost**: Gradient boosting plus performant
- **LightGBM**: Plus rapide, mêmes performances
- **CatBoost**: Gère natif les catégorielles
- **Neural Networks**: Pour relations très complexes

4. **Ensemble Methods**:
```python
from sklearn.ensemble import VotingRegressor

ensemble = VotingRegressor([
    ('rf', RandomForestRegressor()),
    ('xgb', XGBRegressor()),
    ('lgbm', LGBMRegressor())
])
```

---

## 🔀 Orchestration avec Airflow

### Architecture Airflow

**Executor**: CeleryExecutor
- Permet l'exécution distribuée des tâches
- Workers peuvent être scalés horizontalement
- Communication via Redis

**Composants**:
1. **Webserver** (Port 8080) - Interface utilisateur
2. **Scheduler** - Planification et déclenchement des tâches
3. **Worker** - Exécution des tâches via Celery
4. **Triggerer** - Gestion des tâches asynchrones
5. **Redis** - Message broker
6. **PostgreSQL** - Metadata database

### DAG: car_price_predictor_pipeline

**Fichier**: `airflow/dags/car_price_ml_pipeline.py`

#### Configuration du DAG

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-science-team',
    'depends_on_past': False,
    'email': ['alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

dag = DAG(
    'car_price_predictor_pipeline',
    default_args=default_args,
    description='Pipeline ML complet pour prédiction prix voitures',
    schedule_interval='@daily',      # Exécution quotidienne à minuit
    start_date=datetime(2026, 2, 1),
    catchup=False,                   # Ne pas rattraper les runs manqués
    max_active_runs=1,               # 1 seul run à la fois
    tags=['ml', 'car-price', 'production']
)
```

#### Architecture des Tâches (7 Tasks)

```
Task 1: check_data_quality
        ↓ (XCom: data_quality)
Task 2: train_model
        ↓ (XCom: run_id)
Task 3: evaluate_model
        ↓ (XCom: evaluation_result)
Task 4: promote_to_staging
        ↓ (XCom: model_version)
Task 5: validate_staging_model
        ↓ (XCom: validation_result)
Task 6: promote_to_production
        ↓ (XCom: production_info)
Task 7: send_pipeline_report
```

### Détail des Tâches

#### Task 1: check_data_quality

**Objectif**: Valider la qualité des données sources avant entraînement

```python
def check_data_quality(**context):
    import pandas as pd
    
    # Charger les données
    df = pd.read_csv('/opt/airflow/project/data/raw/avito_car_dataset_ALL.csv')
    
    # Statistiques de qualité
    data_quality = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_percentage': (df.isnull().sum().sum() / df.size) * 100,
        'duplicates': df.duplicated().sum(),
        'columns': df.columns.tolist(),
        'data_ok': True
    }
    
    # Critères de validation
    if data_quality['total_rows'] < 1000:
        raise ValueError("❌ Trop peu de données (< 1000 lignes)")
    
    if data_quality['missing_percentage'] > 50:
        raise ValueError("❌ Trop de valeurs manquantes (> 50%)")
    
    print(f"✅ Données validées: {data_quality['total_rows']} lignes")
    print(f"📊 Valeurs manquantes: {data_quality['missing_percentage']:.2f}%")
    
    # Push vers XCom
    context['ti'].xcom_push(key='data_quality', value=data_quality)
    return data_quality

task1 = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    dag=dag
)
```

**Checks effectués**:
- ✅ Nombre de lignes suffisant (> 1000)
- ✅ Pourcentage de valeurs manquantes acceptable (< 50%)
- ✅ Présence des colonnes requises
- ✅ Types de données cohérents

**Output XCom**:
```json
{
  "total_rows": 24776,
  "total_columns": 32,
  "missing_percentage": 4.61,
  "duplicates": 276,
  "columns": ["Marque", "Modèle", ...],
  "data_ok": true
}
```

#### Task 2: train_model

**Objectif**: Exécuter le pipeline ML complet et logger dans MLflow

```python
def train_model(**context):
    import sys
    import os
    import mlflow
    import socket
    from mlflow.tracking import MlflowClient
    
    # Ajouter le projet au path
    sys.path.insert(0, '/opt/airflow/project')
    from scripts.train_with_mlflow import CarPricePipeline
    
    # Configuration MLflow avec résolution DNS
    # Solution au problème "Invalid Host header"
    mlflow_uri = f"http://{socket.gethostbyname('mlflow')}:5000"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("car_price_prediction")
    
    # Récupérer les infos de qualité
    data_quality = context['ti'].xcom_pull(
        key='data_quality', 
        task_ids='check_data_quality'
    )
    
    if data_quality:
        print(f"📊 Données validées: {data_quality['total_rows']} lignes")
    
    # Paths absolus (container context)
    project_path = '/opt/airflow/project'
    params_path = os.path.join(project_path, 'params.yaml')
    data_path = os.path.join(project_path, 'data/raw/avito_car_dataset_ALL.csv')
    
    with mlflow.start_run():
        # Initialiser le pipeline avec base_dir
        pipeline = CarPricePipeline(
            params_file=params_path,
            base_dir=project_path
        )
        
        # Exécution du pipeline
        df = pipeline.load_data(filepath=data_path)
        print(f"✅ Données chargées: {df.shape}")
        
        df_clean = pipeline.preprocess_data(df)
        print(f"✅ Données nettoyées: {df_clean.shape}")
        
        X_scaled, y_scaled, feature_names = pipeline.prepare_features(df_clean)
        print(f"✅ Features préparées: {X_scaled.shape}")
        
        X_train, X_test, y_train, y_test = pipeline.train_model(X_scaled, y_scaled)
        print(f"✅ Modèle entraîné: {len(X_train)} train, {len(X_test)} test")
        
        pipeline.save_model()
        print(f"✅ Modèle sauvegardé")
        
        # Récupérer le run_id
        run_id = mlflow.active_run().info.run_id
        
        # Push vers XCom
        context['ti'].xcom_push(key='run_id', value=run_id)
        return run_id

task2 = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)
```

**Points clés**:

1. **DNS Resolution Workaround**:
   - Problème: MLflow rejette `Host: mlflow:5000` (DNS rebinding protection)
   - Solution: Résoudre le nom vers IP: `socket.gethostbyname('mlflow')` → `172.18.0.4`
   - Utiliser: `http://172.18.0.4:5000` au lieu de `http://mlflow:5000`

2. **Paths Absolus**:
   - Container context: `/opt/airflow/project/`
   - Tous les fichiers doivent utiliser des paths absolus
   - Paramètre `base_dir` dans CarPricePipeline

3. **XCom Flow**:
   - Pull: data_quality depuis Task 1
   - Push: run_id pour Task 3

**Output XCom**:
```json
{
  "run_id": "60fca29c15064aa888ceb574de0b0030"
}
```

#### Task 3: evaluate_model

**Objectif**: Récupérer les métriques depuis MLflow et valider la qualité

```python
def evaluate_model(**context):
    import mlflow
    import socket
    from mlflow.tracking import MlflowClient
    
    # Configuration MLflow
    mlflow_uri = f"http://{socket.gethostbyname('mlflow')}:5000"
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()
    
    # Récupérer le run_id depuis Task 2
    run_id = context['ti'].xcom_pull(key='run_id', task_ids='train_model')
    
    if not run_id:
        raise ValueError("❌ Aucun run_id trouvé")
    
    print(f"📊 Évaluation du run: {run_id}")
    
    # Récupérer les métriques depuis MLflow
    run = client.get_run(run_id)
    
    r2_score = run.data.metrics.get('test_r2', 0.0)
    rmse = run.data.metrics.get('test_rmse', 999999)
    mae = run.data.metrics.get('test_mae', 999999)
    
    print(f"📈 R² Score: {r2_score:.4f}")
    print(f"📉 RMSE: {rmse:.4f}")
    print(f"📉 MAE: {mae:.4f}")
    
    # Critères de qualité pour promotion
    is_promotable = (
        r2_score >= 0.70 and    # R² minimum acceptable
        rmse <= 1.0             # RMSE maximum acceptable
    )
    
    evaluation_result = {
        'run_id': run_id,
        'r2_score': r2_score,
        'rmse': rmse,
        'mae': mae,
        'is_promotable': is_promotable,
        'evaluation_date': datetime.now().isoformat()
    }
    
    if is_promotable:
        print("✅ Modèle promotable vers Staging")
    else:
        print("⚠️ Modèle non promotable (critères non satisfaits)")
    
    # Push vers XCom
    context['ti'].xcom_push(key='evaluation_result', value=evaluation_result)
    return evaluation_result

task3 = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag
)
```

**Critères de Promotion**:
- R² ≥ 0.70 (70% de variance expliquée)
- RMSE ≤ 1.0 (erreur raisonnable sur prix normalisés)

**Output XCom**:
```json
{
  "run_id": "60fca29c15064aa888ceb574de0b0030",
  "r2_score": 0.7299,
  "rmse": 0.5188,
  "mae": 0.3872,
  "is_promotable": true,
  "evaluation_date": "2026-02-14T18:07:07.318904"
}
```

#### Task 4: promote_to_staging

**Objectif**: Trouver la version du modèle et la transitionner vers "Staging"

```python
def promote_to_staging(**context):
    import mlflow
    import socket
    from mlflow.tracking import MlflowClient
    
    print("🔄 Promotion vers Staging...")
    
    # Configuration MLflow
    mlflow_uri = f"http://{socket.gethostbyname('mlflow')}:5000"
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()
    
    # Récupérer run_id et evaluation
    run_id = context['ti'].xcom_pull(key='run_id', task_ids='train_model')
    eval_result = context['ti'].xcom_pull(
        key='evaluation_result', 
        task_ids='evaluate_model'
    )
    
    if not eval_result.get('is_promotable', False):
        print("⚠️ Modèle non promotable, skip")
        return "skipped"
    
    model_name = "CarPricePredictor"
    
    # Trouver la version du modèle par run_id
    versions = client.search_model_versions(f"name='{model_name}'")
    
    model_version_obj = None
    for v in versions:
        if v.run_id == run_id:
            model_version_obj = v
            break
    
    if not model_version_obj:
        raise ValueError(f"❌ Aucune version trouvée pour run_id: {run_id}")
    
    # Convertir version en string (XCom serialization)
    version_number = str(model_version_obj.version)
    
    print(f"📦 Promotion de la version {version_number} vers Staging")
    
    # Transition vers Staging
    client.transition_model_version_stage(
        name=model_name,
        version=version_number,
        stage="Staging",
        archive_existing_versions=False  # Garder anciennes versions
    )
    
    print(f"✅ Version {version_number} promue vers Staging")
    
    # Push vers XCom
    context['ti'].xcom_push(key='model_version', value=version_number)
    
    return {
        'model_name': model_name,
        'version': version_number,
        'stage': 'Staging'
    }

task4 = PythonOperator(
    task_id='promote_to_staging',
    python_callable=promote_to_staging,
    dag=dag
)
```

**Points clés**:
- Recherche de la version par `run_id` (pas de re-registration)
- Conversion explicite en string: `str(model_version_obj.version)`
- XCom serialization: Uniquement types simples (str, int, float, dict, list)

**Output XCom**:
```json
{
  "model_version": "1",
  "stage": "Staging"
}
```

#### Task 5: validate_staging_model

**Objectif**: Valider que le modèle est bien en Staging

```python
def validate_staging_model(**context):
    import mlflow
    import socket
    from mlflow.tracking import MlflowClient
    
    print("🧪 Validation du modèle en Staging...")
    
    # Configuration MLflow
    mlflow_uri = f"http://{socket.gethostbyname('mlflow')}:5000"
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()
    
    # Récupérer la version depuis Task 4
    model_version = context['ti'].xcom_pull(
        key='model_version', 
        task_ids='promote_to_staging'
    )
    
    # None check (si promote_to_staging a retourné "skipped")
    if not model_version:
        print("⚠️ Aucune version à valider (modèle non promu)")
        return "skipped"
    
    model_name = "CarPricePredictor"
    
    print(f"📦 Validation de la version {model_version}")
    
    # Récupérer les infos du modèle
    model_info = client.get_model_version(model_name, model_version)
    
    if not model_info:
        raise ValueError(f"❌ Modèle {model_name} v{model_version} introuvable")
    
    print(f"✅ Modèle trouvé: {model_name} v{model_version}")
    print(f"   Stage: {model_info.current_stage}")
    print(f"   Run ID: {model_info.run_id}")
    
    validation_result = {
        'model_name': model_name,
        'model_version': model_version,
        'stage': model_info.current_stage,
        'run_id': model_info.run_id,
        'validation_passed': True
    }
    
    # Push vers XCom
    context['ti'].xcom_push(key='validation_result', value=validation_result)
    
    print("✅ Validation du modèle Staging réussie!")
    return validation_result

task5 = PythonOperator(
    task_id='validate_staging_model',
    python_callable=validate_staging_model,
    dag=dag
)
```

**Validations**:
- ✅ Modèle existe dans MLflow Registry
- ✅ Version correcte
- ✅ Stage = "Staging"
- ✅ Métadonnées cohérentes

**Output XCom**:
```json
{
  "model_name": "CarPricePredictor",
  "model_version": "1",
  "stage": "Staging",
  "run_id": "60fca29c15064aa888ceb574de0b0030",
  "validation_passed": true
}
```

#### Task 6: promote_to_production

**Objectif**: Promouvoir le modèle validé vers "Production"

```python
def promote_to_production(**context):
    import mlflow
    import socket
    from mlflow.tracking import MlflowClient
    
    print("🚀 Promotion du modèle vers Production...")
    
    # Configuration MLflow
    mlflow_uri = f"http://{socket.gethostbyname('mlflow')}:5000"
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()
    
    # Récupérer la version depuis Task 4
    model_version = context['ti'].xcom_pull(
        key='model_version', 
        task_ids='promote_to_staging'
    )
    
    # None check
    if not model_version:
        print("⚠️ Aucune version à promouvoir (modèle non trouvé)")
        return "skipped"
    
    # Vérifier la validation
    validation = context['ti'].xcom_pull(
        key='validation_result', 
        task_ids='validate_staging_model'
    )
    
    if validation and not validation.get('validation_passed', True):
        print("⚠️ Validation échouée, promotion vers Production annulée")
        return "skipped"
    
    model_name = "CarPricePredictor"
    
    print(f"🎯 Promotion de la version {model_version} vers Production")
    
    try:
        # Transition vers Production
        # archive_existing_versions=True → Archive l'ancien modèle Production
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Production",
            archive_existing_versions=True
        )
        
        print(f"✅ Version {model_version} promue vers Production")
        print("📦 Ancien modèle Production archivé")
        
        production_info = {
            'model_name': model_name,
            'version': model_version,
            'stage': 'Production',
            'promoted_at': datetime.now().isoformat()
        }
        
        # Push vers XCom
        context['ti'].xcom_push(key='production_info', value=production_info)
        
        return production_info
        
    except Exception as e:
        print(f"❌ Erreur lors de la promotion: {str(e)}")
        raise

task6 = PythonOperator(
    task_id='promote_to_production',
    python_callable=promote_to_production,
    dag=dag
)
```

**Comportement**:
- Archive automatiquement l'ancien modèle en Production
- Seul le nouveau modèle reste en stage "Production"
- Anciens modèles passent en "Archived"

**Output XCom**:
```json
{
  "model_name": "CarPricePredictor",
  "version": "1",
  "stage": "Production",
  "promoted_at": "2026-02-14T18:07:10.123456"
}
```

#### Task 7: send_pipeline_report

**Objectif**: Générer un rapport JSON complet du pipeline

```python
def send_pipeline_report(**context):
    import json
    import numpy as np
    from datetime import datetime
    
    print("📧 Génération du rapport de pipeline...")
    
    def convert_to_json_serializable(obj):
        """Convertir les types numpy en types Python natifs"""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    # Récupérer toutes les informations des tâches précédentes
    data_quality = context['ti'].xcom_pull(
        key='data_quality', 
        task_ids='check_data_quality'
    )
    run_id = context['ti'].xcom_pull(
        key='run_id', 
        task_ids='train_model'
    )
    eval_result = context['ti'].xcom_pull(
        key='evaluation_result', 
        task_ids='evaluate_model'
    )
    validation = context['ti'].xcom_pull(
        key='validation_result', 
        task_ids='validate_staging_model'
    )
    production = context['ti'].xcom_pull(
        key='production_info', 
        task_ids='promote_to_production'
    )
    
    # Créer le rapport avec conversion des types numpy
    report = convert_to_json_serializable({
        'pipeline_date': datetime.now().isoformat(),
        'pipeline_id': context['dag_run'].run_id,
        'data_quality': data_quality,
        'training': {
            'run_id': run_id
        },
        'evaluation': eval_result,
        'staging': validation,
        'production': production,
        'status': 'SUCCESS'
    })
    
    # Sauvegarder le rapport
    project_path = '/opt/airflow/project'
    report_path = os.path.join(
        project_path, 
        'reports', 
        f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Rapport sauvegardé: {report_path}")
    print(f"📊 Status: {report['status']}")
    
    if eval_result and 'r2_score' in eval_result:
        print(f"📊 R² Score: {eval_result['r2_score']:.4f}")
    
    return report

task7 = PythonOperator(
    task_id='send_pipeline_report',
    python_callable=send_pipeline_report,
    trigger_rule='all_done',  # Exécute même si tâches précédentes échouent
    dag=dag
)
```

**Points clés**:

1. **Conversion numpy → JSON**:
   - Problème: `np.bool_`, `np.int64`, `np.float64` non sérialisables en JSON
   - Solution: Fonction récursive `convert_to_json_serializable()`
   - Convertit tous les types numpy en types Python natifs

2. **trigger_rule='all_done'**:
   - Exécute cette tâche même si des tâches précédentes ont échoué
   - Permet de toujours avoir un rapport, même en cas d'échec partiel

**Output**: Fichier JSON dans `reports/pipeline_report_YYYYMMDD_HHMMSS.json`

### Dépendances des Tâches

```python
# Définir les dépendances (ordre d'exécution)
task1 >> task2 >> task3 >> task4 >> task5 >> task6 >> task7
```

**Équivalent à**:
```python
task1.set_downstream(task2)
task2.set_downstream(task3)
task3.set_downstream(task4)
task4.set_downstream(task5)
task5.set_downstream(task6)
task6.set_downstream(task7)
```

### XCom (Cross-Communication)

**Flux de données entre tâches**:

```
Task 1 → data_quality → Task 2
Task 2 → run_id → Task 3, Task 4
Task 3 → evaluation_result → Task 4
Task 4 → model_version → Task 5, Task 6
Task 5 → validation_result → Task 6
Task 6 → production_info → Task 7
Toutes → Task 7 (agrégation)
```

**Limitations XCom**:
- Taille max: ~48KB (limite PostgreSQL)
- Types supportés: str, int, float, dict, list, bool
- Pas supportés: numpy types, objets custom, fichiers

**Best Practices**:
- Push uniquement des métadonnées (IDs, metrics)
- Pas de DataFrames ou modèles (utiliser le filesystem)
- Toujours vérifier None avant utilisation
- Convertir types numpy avant push

### Monitoring du Pipeline

#### Via Airflow UI

1. **Grid View** - Historique des runs
2. **Graph View** - Visualisation des dépendances
3. **Gantt View** - Timeline d'exécution
4. **Task Duration** - Performance par tâche
5. **Logs** - Logs détaillés de chaque tâche

#### Logs Airflow

```bash
# Logs d'une tâche spécifique
tail -f airflow/logs/dag_id=car_price_predictor_pipeline/\
run_id=manual__2026-02-14T18:00:09+00:00/\
task_id=train_model/attempt=1.log
```

#### Alertes Email

Configuration dans `default_args`:
```python
'email': ['alerts@example.com'],
'email_on_failure': True,   # Email si échec
'email_on_retry': False,    # Pas d'email lors des retries
```

**Configuration SMTP** (dans `airflow.cfg` ou env vars):
```ini
[smtp]
smtp_host = smtp.gmail.com
smtp_starttls = True
smtp_ssl = False
smtp_user = your-email@gmail.com
smtp_password = your-app-password
smtp_port = 587
smtp_mail_from = airflow@example.com
```

---

## 📊 Tracking avec MLflow

### Architecture MLflow

```
MLflow Server (http://172.18.0.4:5000)
    ↓
PostgreSQL (Backend Store)
    - Experiments
    - Runs
    - Parameters
    - Metrics
    - Tags
    ↓
File System (Artifact Store)
    - Models (.pkl)
    - Plots (.png)
    - CSV files
    - JSON metadata
```

### Experiment: car_price_prediction

```python
import mlflow

# Configuration
mlflow.set_tracking_uri("http://172.18.0.4:5000")
mlflow.set_experiment("car_price_prediction")

# Démarrer un run
with mlflow.start_run(run_name="training_2026-02-14"):
    
    # 1. Log des paramètres
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 20)
    mlflow.log_param("min_samples_split", 5)
    mlflow.log_param("min_samples_leaf", 2)
    mlflow.log_param("max_features", "sqrt")
    mlflow.log_param("train_samples", 6019)
    mlflow.log_param("test_samples", 1505)
    mlflow.log_param("n_features", 27)
    
    # 2. Log des métriques
    mlflow.log_metric("train_r2", 0.8689)
    mlflow.log_metric("test_r2", 0.7299)
    mlflow.log_metric("test_rmse", 0.5188)
    mlflow.log_metric("test_mae", 0.3872)
    mlflow.log_metric("train_test_gap", 0.1390)  # Overfitting indicator
    
    # 3. Log des artifacts (fichiers)
    mlflow.log_artifact("artifacts/feature_importance.png")
    mlflow.log_artifact("artifacts/predictions_plot.png")
    mlflow.log_artifact("artifacts/residuals_plot.png")
    mlflow.log_artifact("artifacts/feature_importance.csv")
    mlflow.log_artifact("artifacts/feature_info.json")
    mlflow.log_artifact("artifacts/price_scaler_info.json")
    
    # 4. Log du modèle avec auto-logging
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="CarPricePredictor",
        conda_env={
            'name': 'car_price_env',
            'channels': ['defaults', 'conda-forge'],
            'dependencies': [
                'python=3.12',
                'scikit-learn=1.3.2',
                'pandas=2.1.4',
                'numpy=1.26.2'
            ]
        },
        pip_requirements=[
            'scikit-learn==1.3.2',
            'pandas==2.1.4',
            'numpy==1.26.2',
            'joblib==1.3.2'
        ]
    )
    
    # 5. Log des tags
    mlflow.set_tag("model_type", "RandomForestRegressor")
    mlflow.set_tag("dataset", "avito_car_dataset_ALL")
    mlflow.set_tag("feature_engineering", "label_encoding")
    mlflow.set_tag("scaling", "StandardScaler")
    mlflow.set_tag("environment", "production")
    
    # Récupérer le run_id
    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")
```

### Model Registry

#### Enregistrement Initial

Lors du premier `log_model` avec `registered_model_name`:
- Crée le modèle "CarPricePredictor" dans le Registry
- Version 1, stage "None"

#### Transitions de Stages

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Trouver la version par run_id
versions = client.search_model_versions("name='CarPricePredictor'")
model_version = None
for v in versions:
    if v.run_id == run_id:
        model_version = v
        break

version_number = str(model_version.version)

# 1. None → Staging
client.transition_model_version_stage(
    name="CarPricePredictor",
    version=version_number,
    stage="Staging",
    archive_existing_versions=False
)

# 2. Staging → Production
client.transition_model_version_stage(
    name="CarPricePredictor",
    version=version_number,
    stage="Production",
    archive_existing_versions=True  # Archive l'ancien Production
)

# 3. Production → Archived (manuel ou automatique)
client.transition_model_version_stage(
    name="CarPricePredictor",
    version="1",
    stage="Archived",
    archive_existing_versions=False
)
```

#### Lifecycle des Modèles

```
Nouveau Modèle
    ↓
Stage: None (enregistré dans Registry)
    ↓ (si metrics OK)
Stage: Staging (en test)
    ↓ (si validation OK)
Stage: Production (déployé)
    ↓ (remplacé par nouveau)
Stage: Archived (ancien modèle)
```

#### Chargement d'un Modèle

```python
import mlflow

# Option 1: Par version
model = mlflow.pyfunc.load_model("models:/CarPricePredictor/1")

# Option 2: Par stage
model = mlflow.pyfunc.load_model("models:/CarPricePredictor/Production")

# Option 3: Par run_id
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

# Utilisation
predictions = model.predict(X_new)
```

### Artifacts Structure

```
mlflow/mlruns/
└── <experiment_id>/
    └── <run_id>/
        ├── artifacts/
        │   ├── model/
        │   │   ├── MLmodel                      # Métadonnées MLflow
        │   │   ├── conda.yaml                   # Env Conda
        │   │   ├── requirements.txt             # Dépendances pip
        │   │   ├── python_env.yaml              # Env Python
        │   │   ├── model.pkl                    # Modèle sérialisé
        │   │   └── input_example.json           # Exemple d'input
        │   ├── feature_importance.png
        │   ├── predictions_plot.png
        │   ├── residuals_plot.png
        │   ├── feature_importance.csv
        │   ├── feature_info.json
        │   └── price_scaler_info.json
        ├── metrics/
        │   ├── train_r2                         # 0.8689
        │   ├── test_r2                          # 0.7299
        │   ├── test_rmse                        # 0.5188
        │   └── test_mae                         # 0.3872
        ├── params/
        │   ├── n_estimators                     # 100
        │   ├── max_depth                        # 20
        │   ├── min_samples_split                # 5
        │   └── ...
        └── tags/
            ├── mlflow.runName                   # "training_2026-02-14"
            ├── mlflow.user                      # "airflow"
            ├── model_type                       # "RandomForestRegressor"
            └── ...
```

### Interface MLflow UI

#### Experiments View
- Liste des runs d'expérience
- Comparaison des métriques
- Filtre par params/metrics
- Tri par performance

#### Models View (Registry)
- Liste des modèles enregistrés
- Versions et stages
- Transitions history
- Model card et description

#### Run Detail
- Tous les params loggés
- Toutes les métriques
- Artifacts téléchargeables
- Code source (git)
- Environment info

---

## 🌐 Services et Ports

### Stack Docker Complète

| Service | Container Name | Port(s) | URL | Credentials |
|---------|---------------|---------|-----|-------------|
| **Airflow Webserver** | airflow-webserver | 8080 | http://localhost:8080 | admin / airflow |
| **Airflow Scheduler** | airflow-scheduler | - | - | - |
| **Airflow Worker** | airflow-worker | - | - | - |
| **Airflow Triggerer** | airflow-triggerer | - | - | - |
| **MLflow Server** | mlflow-server | 5050 (external)<br>5000 (internal) | http://localhost:5050 | - |
| **Streamlit App** | streamlit-app | 8501 | http://localhost:8501 | - |
| **PostgreSQL (Airflow)** | postgres-airflow | 54322 | localhost:54322 | airflow / airflow |
| **PostgreSQL (MLflow)** | postgres-mlflow | 54323 | localhost:54323 | mlflow / mlflow |
| **Redis** | redis | 6379 | localhost:6379 | - |

### Network Configuration

**Bridge Network**: `mlops-network`

```yaml
networks:
  mlops-network:
    driver: bridge
```

**DNS Resolution Interne**:
- Services communiquent via noms: `mlflow`, `postgres-airflow`, `redis`
- Exemple: `http://mlflow:5000` (depuis containers)
- Résolution externe: `http://localhost:5050` (depuis host)

**Workaround DNS Issue**:
```python
import socket

# Résoudre nom → IP pour éviter "Invalid Host header"
mlflow_ip = socket.gethostbyname('mlflow')  # → 172.18.0.4
mlflow_uri = f"http://{mlflow_ip}:5000"
```

### Health Checks

#### Via CLI

```bash
# Tous les services
docker-compose -f docker-compose-full.yml ps

# Health check Airflow
curl http://localhost:8080/health

# Health check MLflow
curl http://localhost:5050/health

# Health check Redis
docker-compose -f docker-compose-full.yml exec redis redis-cli ping
# Réponse: PONG

# Health check PostgreSQL
docker-compose -f docker-compose-full.yml exec postgres-airflow \
    psql -U airflow -d airflow -c "SELECT 1;"
```

#### Via Docker

```bash
# Status détaillé
docker-compose -f docker-compose-full.yml ps --format json | jq .

# Logs en temps réel
docker-compose -f docker-compose-full.yml logs -f --tail=50

# Logs d'un service
docker-compose -f docker-compose-full.yml logs -f mlflow-server
```

### Volumes Docker

```yaml
volumes:
  # Données persistantes
  postgres-db-volume:     # Airflow metadata
  postgres-mlflow-volume: # MLflow tracking
  
  # Montages du projet
  ./airflow/dags:/opt/airflow/dags
  ./airflow/logs:/opt/airflow/logs
  ./airflow/config:/opt/airflow/config
  .:/opt/airflow/project  # Projet complet
```

---

## 💻 Utilisation

### Démarrage Rapide

#### 1. Démarrer la Stack

```bash
# Donner les permissions
chmod +x docker-start-full.sh docker-stop-full.sh docker-reset-full.sh

# Démarrer tous les services
./docker-start-full.sh

# Attendre 2-3 minutes
# Suivre les logs
docker-compose -f docker-compose-full.yml logs -f
```

#### 2. Accéder aux Interfaces

- **Airflow**: http://localhost:8080 (admin / airflow)
- **MLflow**: http://localhost:5050
- **Streamlit**: http://localhost:8501

#### 3. Lancer le Pipeline ML

**Via Airflow UI**:
1. Aller sur http://localhost:8080
2. Connexion: admin / airflow
3. Activer le DAG `car_price_predictor_pipeline` (toggle)
4. Cliquer sur "Trigger DAG" (bouton play ▶️)
5. Suivre l'exécution dans Grid View

**Via CLI**:
```bash
# Trigger manuel
docker-compose -f docker-compose-full.yml exec airflow-webserver \
    airflow dags trigger car_price_predictor_pipeline

# Voir le statut
docker-compose -f docker-compose-full.yml exec airflow-webserver \
    airflow dags list

# Voir les runs
docker-compose -f docker-compose-full.yml exec airflow-webserver \
    airflow dags list-runs -d car_price_predictor_pipeline
```

### Utilisation Avancée

#### Tester une Tâche Individuellement

```bash
# Syntaxe: airflow tasks test <dag_id> <task_id> <execution_date>

# Tester check_data_quality
docker-compose -f docker-compose-full.yml exec airflow-webserver \
    airflow tasks test car_price_predictor_pipeline check_data_quality 2026-02-14

# Tester train_model
docker-compose -f docker-compose-full.yml exec airflow-webserver \
    airflow tasks test car_price_predictor_pipeline train_model 2026-02-14
```

#### Clear une Tâche (Réexécuter)

```bash
# Clear une tâche spécifique
docker-compose -f docker-compose-full.yml exec airflow-webserver \
    airflow tasks clear car_price_predictor_pipeline -t train_model

# Clear tout le DAG
docker-compose -f docker-compose-full.yml exec airflow-webserver \
    airflow dags backfill car_price_predictor_pipeline \
    -s 2026-02-14 -e 2026-02-14
```

#### Modifier le DAG

```bash
# Editer le DAG
nano airflow/dags/car_price_ml_pipeline.py

# Airflow détecte automatiquement les changements (30s)
# Ou forcer le refresh
docker-compose -f docker-compose-full.yml restart airflow-scheduler
```

### Utilisation de l'App Streamlit

#### Accès

http://localhost:8501

#### Fonctionnalités

1. **Formulaire de Prédiction**:
   - Sélectionner Marque, Modèle, Année
   - Entrer Kilométrage, Puissance fiscale
   - Choisir Type de carburant, Transmission
   - Cocher les équipements

2. **Prédiction en Temps Réel**:
   - Cliquer sur "Prédire le Prix"
   - Obtenir le prix estimé en DH

3. **Visualisations**:
   - Feature importance
   - Historique des prédictions
   - Distribution des prix

#### Code d'Utilisation

```python
import streamlit as st
import joblib
import pandas as pd

# Charger le modèle
model = joblib.load('models/car_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Interface
st.title("🚗 Prédicteur de Prix de Voiture")

# Inputs
marque = st.selectbox("Marque", ['Dacia', 'Renault', 'Peugeot', ...])
modele = st.selectbox("Modèle", [...])
annee = st.slider("Année", 2000, 2024, 2020)
km = st.number_input("Kilométrage", 0, 500000, 50000)

# Prédiction
if st.button("Prédire"):
    # Preprocessing
    X = prepare_features(marque, modele, annee, km, ...)
    X_scaled = scaler.transform(X)
    
    # Prédiction
    prediction = model.predict(X_scaled)
    prix_predit = inverse_transform_price(prediction)
    
    st.success(f"Prix estimé: {prix_predit:,.0f} DH")
```

### Accès aux Données

#### Rapports JSON

```bash
# Dernier rapport
cat reports/pipeline_report_*.json | tail -1 | jq .

# Tous les rapports
ls -lh reports/

# Rechercher un run spécifique
grep -r "run_id" reports/ | grep "60fca29c"
```

#### Artifacts MLflow

```bash
# Liste des artifacts
ls -lh artifacts/

# Ouvrir les graphiques
open artifacts/feature_importance.png
open artifacts/predictions_plot.png
open artifacts/residuals_plot.png

# Lire les CSV
cat artifacts/feature_importance.csv | column -t -s,

# Lire les JSON
cat artifacts/feature_info.json | jq .
```

#### Modèles

```bash
# Modèles sauvegardés
ls -lh models/

# Taille du modèle
du -sh models/car_model.pkl

# Inspecter le modèle (Python)
python -c "
import joblib
model = joblib.load('models/car_model.pkl')
print(f'Type: {type(model)}')
print(f'N estimators: {model.n_estimators}')
print(f'Max depth: {model.max_depth}')
"
```

#### Logs Airflow

```bash
# Structure des logs
tree airflow/logs/ -L 3

# Logs d'un run spécifique
tail -f airflow/logs/dag_id=car_price_predictor_pipeline/\
run_id=manual__2026-02-14T18:00:09+00:00/\
task_id=train_model/attempt=1.log

# Rechercher une erreur
grep -r "ERROR" airflow/logs/dag_id=car_price_predictor_pipeline/

# Derniers logs de toutes les tâches
find airflow/logs/dag_id=car_price_predictor_pipeline/ \
    -name "attempt=1.log" -exec tail -20 {} \; -print
```

---

## 📈 Monitoring et Reporting

### Pipeline Report Structure

Chaque exécution génère un rapport JSON complet:

```json
{
  "pipeline_date": "2026-02-14T18:07:11.823224",
  "pipeline_id": "manual__2026-02-14T18:06:59+00:00",
  "data_quality": {
    "total_rows": 24776,
    "total_columns": 32,
    "missing_percentage": 4.606,
    "duplicates": 276,
    "columns": ["Marque", "Modèle", "Année-Modèle", ...],
    "data_ok": true
  },
  "training": {
    "run_id": "60fca29c15064aa888ceb574de0b0030",
    "samples_loaded": 24776,
    "samples_cleaned": 7524,
    "train_samples": 6019,
    "test_samples": 1505,
    "n_features": 27
  },
  "evaluation": {
    "run_id": "60fca29c15064aa888ceb574de0b0030",
    "r2_score": 0.7299,
    "rmse": 0.5188,
    "mae": 0.3872,
    "is_promotable": true,
    "evaluation_date": "2026-02-14T18:07:07.318904"
  },
  "staging": {
    "model_name": "CarPricePredictor",
    "version": "1",
    "stage": "Staging",
    "run_id": "60fca29c15064aa888ceb574de0b0030",
    "validation_passed": true
  },
  "production": {
    "model_name": "CarPricePredictor",
    "version": "1",
    "stage": "Production",
    "promoted_at": "2026-02-14T18:07:10.123456",
    "archived_previous": true
  },
  "status": "SUCCESS",
  "execution_time_seconds": 45.3,
  "tasks_completed": 7,
  "tasks_failed": 0
}
```

### Métriques Clés à Surveiller

#### 1. Data Quality
- ✅ `total_rows` ≥ 1000 (volume suffisant)
- ✅ `missing_percentage` ≤ 50% (qualité acceptable)
- ✅ `duplicates` (à minimiser)

#### 2. Model Performance
- ✅ `test_r2` ≥ 0.70 (70% variance expliquée)
- ✅ `test_rmse` ≤ 1.0 (erreur raisonnable)
- ✅ `test_mae` ≤ 0.5 (erreur absolue faible)
- ⚠️ `train_test_gap` < 0.15 (pas trop d'overfitting)

#### 3. Pipeline Health
- ✅ `status` = "SUCCESS"
- ✅ `tasks_completed` = 7
- ✅ `tasks_failed` = 0
- ⏱️ `execution_time_seconds` < 600 (< 10 min)

### Alertes et Notifications

#### Email Alerts (Configuration)

```python
# Dans docker-compose-full.yml ou .env
AIRFLOW__SMTP__SMTP_HOST=smtp.gmail.com
AIRFLOW__SMTP__SMTP_PORT=587
AIRFLOW__SMTP__SMTP_USER=your-email@gmail.com
AIRFLOW__SMTP__SMTP_PASSWORD=your-app-password
AIRFLOW__SMTP__SMTP_MAIL_FROM=airflow@example.com
```

```python
# Dans le DAG
default_args = {
    'email': ['team@example.com', 'alerts@example.com'],
    'email_on_failure': True,    # Email si échec
    'email_on_retry': False,     # Pas d'email lors retries
    'email_on_success': False,   # Optionnel: email si succès
}
```

#### Slack Notifications (Optionnel)

```python
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

def send_slack_alert(context):
    """Callback en cas d'échec"""
    slack_alert = SlackWebhookOperator(
        task_id='slack_alert',
        http_conn_id='slack_webhook',
        message=f"""
        ❌ *Pipeline Failed*
        
        DAG: {context['dag'].dag_id}
        Task: {context['task'].task_id}
        Execution Date: {context['execution_date']}
        Log: {context['task_instance'].log_url}
        """,
        channel='#ml-alerts'
    )
    return slack_alert.execute(context=context)

# Dans le DAG
default_args = {
    'on_failure_callback': send_slack_alert
}
```

### Dashboard Monitoring

#### Airflow Built-in

**Grid View**:
- Historique des runs (vert = succès, rouge = échec)
- Vue d'ensemble rapide
- Drill-down par task

**Graph View**:
- Visualisation des dépendances
- Status de chaque tâche
- Flowchart du pipeline

**Gantt View**:
- Timeline d'exécution
- Durée de chaque tâche
- Identification des bottlenecks

**Task Duration**:
- Durée moyenne par tâche
- Tendances temporelles
- Outliers

#### MLflow UI

**Experiments**:
- Tableau comparatif des runs
- Tri par métrique (R², RMSE, MAE)
- Filtre par paramètre
- Visualisation des métriques

**Models Registry**:
- Lifecycle des modèles
- Versions et stages
- Transitions history
- Model card

#### Custom Dashboard (Optionnel)

**Grafana + Prometheus**:

```yaml
# docker-compose monitoring
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
```

**Métriques à collecter**:
- Airflow task success rate
- Average execution time
- MLflow run count
- Model performance over time
- Data quality metrics

---

## 🔧 Troubleshooting

### Problèmes Fréquents

#### 1. Services ne démarrent pas

**Symptôme**: `docker-compose ps` montre des services "Exited" ou "unhealthy"

**Diagnostic**:
```bash
# Vérifier les logs
docker-compose -f docker-compose-full.yml logs airflow-init
docker-compose -f docker-compose-full.yml logs postgres-airflow
docker-compose -f docker-compose-full.yml logs mlflow-server

# Vérifier les resources
docker stats

# Vérifier l'espace disque
df -h
```

**Solutions**:

a) **Permissions insuffisantes**:
```bash
# Fix permissions
chmod -R 777 airflow/logs
chmod -R 777 data
chmod -R 777 models
chmod -R 777 artifacts
chmod -R 777 reports

# Ou chown avec votre user
sudo chown -R $(whoami):$(whoami) airflow/ data/ models/ artifacts/ reports/
```

b) **Ports déjà utilisés**:
```bash
# Vérifier les ports
lsof -i :8080  # Airflow
lsof -i :5050  # MLflow
lsof -i :54322 # PostgreSQL Airflow
lsof -i :54323 # PostgreSQL MLflow

# Tuer le processus
kill -9 <PID>

# Ou changer le port dans docker-compose-full.yml
```

c) **Mémoire insuffisante**:
```bash
# Vérifier la mémoire disponible
free -h

# Allouer plus de mémoire à Docker Desktop
# Préférences → Resources → Memory (minimum 8GB)
```

d) **Reset complet**:
```bash
# Arrêter et supprimer tout
./docker-reset-full.sh

# Redémarrer proprement
./docker-start-full.sh
```

#### 2. DAG non visible dans Airflow

**Symptôme**: Le DAG `car_price_predictor_pipeline` n'apparaît pas dans l'UI

**Diagnostic**:
```bash
# Vérifier que le fichier est monté
docker-compose -f docker-compose-full.yml exec airflow-webserver \
    ls -lh /opt/airflow/dags/

# Vérifier les erreurs de parsing
docker-compose -f docker-compose-full.yml exec airflow-webserver \
    airflow dags list

# Logs du scheduler
docker-compose -f docker-compose-full.yml logs -f airflow-scheduler
```

**Solutions**:

a) **Erreur de syntaxe Python**:
```bash
# Valider la syntaxe
python -m py_compile airflow/dags/car_price_ml_pipeline.py

# Ou dans le container
docker-compose -f docker-compose-full.yml exec airflow-webserver \
    python -m py_compile /opt/airflow/dags/car_price_ml_pipeline.py
```

b) **Imports manquants**:
```bash
# Vérifier les imports
docker-compose -f docker-compose-full.yml exec airflow-webserver \
    python -c "
import sys
sys.path.insert(0, '/opt/airflow/project')
from scripts.train_with_mlflow import CarPricePipeline
print('OK')
"
```

c) **Redémarrer le scheduler**:
```bash
docker-compose -f docker-compose-full.yml restart airflow-scheduler
```

#### 3. Erreur MLflow "Invalid Host header"

**Symptôme**: 
```
mlflow.exceptions.MlflowException: API request to endpoint /api/2.0/mlflow/runs/create failed with error code 403 != 200. Response body: 'Invalid Host header'
```

**Cause**: Werkzeug/Flask security check rejette les requêtes avec Host header = nom de domaine Docker

**Solution**: Résoudre DNS vers IP

```python
import socket

# Au lieu de
mlflow.set_tracking_uri("http://mlflow:5000")  # ❌

# Utiliser
mlflow_ip = socket.gethostbyname('mlflow')  # → 172.18.0.4
mlflow_uri = f"http://{mlflow_ip}:5000"
mlflow.set_tracking_uri(mlflow_uri)  # ✅
```

**Vérification**:
```bash
# Depuis un container Airflow
docker-compose -f docker-compose-full.yml exec airflow-webserver \
    python -c "
import socket
print(socket.gethostbyname('mlflow'))
"

# Test de connexion
docker-compose -f docker-compose-full.yml exec airflow-webserver \
    curl http://172.18.0.4:5000/health
```

#### 4. Tâche Airflow en échec constant

**Symptôme**: Une tâche fail systématiquement avec le même erreur

**Diagnostic**:
```bash
# Voir les logs détaillés
docker-compose -f docker-compose-full.yml exec airflow-webserver \
    airflow tasks test car_price_predictor_pipeline train_model 2026-02-14

# Ou via l'UI
# Cliquer sur la tâche → View Log
```

**Solutions courantes**:

a) **FileNotFoundError**:
```python
# Utiliser des paths absolus
data_path = '/opt/airflow/project/data/raw/avito_car_dataset_ALL.csv'
# Au lieu de
data_path = 'data/raw/avito_car_dataset_ALL.csv'  # ❌
```

b) **ModuleNotFoundError**:
```bash
# Vérifier les dépendances dans le container
docker-compose -f docker-compose-full.yml exec airflow-webserver \
    pip list | grep mlflow

# Rebuilder si nécessaire
docker-compose -f docker-compose-full.yml build airflow-webserver
docker-compose -f docker-compose-full.yml up -d
```

c) **XCom TypeError**:
```python
# Toujours vérifier None
data = ti.xcom_pull(key='data', task_ids='previous_task')
if not data:
    print("⚠️ No data from previous task")
    return "skipped"

# Convertir types numpy avant push
import numpy as np
value = int(np_value) if isinstance(np_value, np.integer) else np_value
ti.xcom_push(key='value', value=value)
```

#### 5. Base de données PostgreSQL corrompue

**Symptôme**: Airflow ou MLflow ne démarre pas, erreurs de connexion DB

**Diagnostic**:
```bash
# Vérifier les logs PostgreSQL
docker-compose -f docker-compose-full.yml logs postgres-airflow
docker-compose -f docker-compose-full.yml logs postgres-mlflow

# Tester la connexion
docker-compose -f docker-compose-full.yml exec postgres-airflow \
    psql -U airflow -d airflow -c "SELECT version();"
```

**Solution**: Reset complet

```bash
# Arrêter tout
docker-compose -f docker-compose-full.yml down

# Supprimer les volumes
docker volume rm pythonproject9_postgres-db-volume
docker volume rm pythonproject9_postgres-mlflow-volume

# Ou supprimer les dossiers locaux
rm -rf postgres-db postgres-mlflow

# Redémarrer
docker-compose -f docker-compose-full.yml up -d
```

#### 6. Modèle non trouvé dans MLflow Registry

**Symptôme**: `get_model_version` échoue avec "Model not found"

**Diagnostic**:
```bash
# Lister les modèles enregistrés
docker-compose -f docker-compose-full.yml exec airflow-webserver \
    python -c "
from mlflow.tracking import MlflowClient
import socket

mlflow_uri = f'http://{socket.gethostbyname(\"mlflow\")}:5000'
client = MlflowClient(tracking_uri=mlflow_uri)

models = client.search_registered_models()
for m in models:
    print(f'Model: {m.name}')
    versions = client.search_model_versions(f'name=\"{m.name}\"')
    for v in versions:
        print(f'  Version {v.version}: stage={v.current_stage}, run_id={v.run_id}')
"
```

**Solutions**:

a) **Modèle pas encore enregistré**:
- Attendre que `train_model` termine
- Vérifier les logs de `train_model`
- S'assurer que `mlflow.sklearn.log_model` avec `registered_model_name` est appelé

b) **Mauvais nom de modèle**:
```python
# Vérifier le nom exact
model_name = "CarPricePredictor"  # Sensible à la casse
```

c) **Re-enregistrer le modèle**:
```python
import mlflow

mlflow.set_tracking_uri(...)
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri, "CarPricePredictor")
```

### Debug Mode

#### Activer les Logs Détaillés

**Airflow**:
```yaml
# docker-compose-full.yml
environment:
  AIRFLOW__LOGGING__LOGGING_LEVEL: DEBUG  # Au lieu de INFO
```

**MLflow**:
```yaml
# docker-compose-full.yml
services:
  mlflow:
    command: >
      mlflow server
      --backend-store-uri postgresql://...
      --default-artifact-root /mlflow/artifacts
      --host 0.0.0.0
      --port 5000
      --gunicorn-opts "--log-level=debug"  # Ajouter
```

#### Exécution Manuelle Python

```bash
# Entrer dans le container
docker-compose -f docker-compose-full.yml exec airflow-webserver bash

# Exécuter le script directement
cd /opt/airflow/project
python -c "
from scripts.train_with_mlflow import CarPricePipeline

pipeline = CarPricePipeline(
    params_file='params.yaml',
    base_dir='/opt/airflow/project'
)

df = pipeline.load_data('data/raw/avito_car_dataset_ALL.csv')
print(f'Loaded: {df.shape}')

df_clean = pipeline.preprocess_data(df)
print(f'Cleaned: {df_clean.shape}')
"
```

#### Tester une Fonction Spécifique

```python
# Test unitaire ad-hoc
docker-compose -f docker-compose-full.yml exec airflow-webserver python << 'EOF'
import sys
sys.path.insert(0, '/opt/airflow/project')

def test_check_data_quality():
    import pandas as pd
    df = pd.read_csv('/opt/airflow/project/data/raw/avito_car_dataset_ALL.csv')
    
    assert len(df) > 1000, "Too few rows"
    assert df.isnull().sum().sum() / df.size < 0.5, "Too many missing"
    
    print("✅ All checks passed")

test_check_data_quality()
EOF
```

### Performance Optimization

#### 1. Augmenter les Ressources Docker

```yaml
# docker-compose-full.yml
x-airflow-common:
  &airflow-common
  deploy:
    resources:
      limits:
        cpus: '2.0'
        memory: 4G
      reservations:
        cpus: '1.0'
        memory: 2G
```

#### 2. Optimiser le Modèle

```python
# Réduire n_estimators si trop lent
model = RandomForestRegressor(
    n_estimators=50,  # Au lieu de 100
    n_jobs=-1  # Utiliser tous les CPU
)

# Ou utiliser un modèle plus rapide
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor()
```

#### 3. Paralléliser les Tâches Airflow

```python
# Augmenter le parallélisme
# airflow.cfg ou env vars
AIRFLOW__CORE__PARALLELISM=32
AIRFLOW__CORE__DAG_CONCURRENCY=16
AIRFLOW__CORE__MAX_ACTIVE_RUNS_PER_DAG=3
```

#### 4. Caching Docker Layers

```dockerfile
# Dockerfile.airflow
# Copier requirements AVANT le code (meilleur caching)
COPY requirements-airflow.txt .
RUN pip install --no-cache-dir -r requirements-airflow.txt

# Copier le code en dernier
COPY . /opt/airflow/project/
```

---

## 🚀 Déploiement en Production

### Checklist Production

#### Sécurité
- [ ] Changer les mots de passe par défaut
  - Airflow admin / airflow → Mot de passe fort
  - PostgreSQL airflow / airflow → Mot de passe fort
  - PostgreSQL mlflow / mlflow → Mot de passe fort
- [ ] Activer HTTPS (SSL/TLS)
  - Nginx reverse proxy
  - Certificats Let's Encrypt
- [ ] Authentification forte
  - OAuth2 (Google, GitHub)
  - LDAP
  - SSO
- [ ] Secrets management
  - AWS Secrets Manager
  - HashiCorp Vault
  - Docker Secrets
- [ ] Network isolation
  - VPC privé
  - Security groups
  - Firewall rules

#### Infrastructure
- [ ] Haute disponibilité
  - Multiple workers Airflow (≥2)
  - PostgreSQL en mode HA (replication)
  - Redis Sentinel ou Cluster
- [ ] Backup automatique
  - PostgreSQL dumps quotidiens
  - Artifacts MLflow S3/GCS
  - Configuration as code (Git)
- [ ] Monitoring
  - Prometheus + Grafana
  - Alerting (PagerDuty, OpsGenie)
  - Logs centralisés (ELK, Splunk)
- [ ] Auto-scaling
  - Kubernetes HPA
  - AWS Auto Scaling
  - GCP MIG

#### Performance
- [ ] Resource limits
  - CPU/RAM par container
  - PostgreSQL tuning
  - Redis memory limit
- [ ] Caching
  - Redis pour XCom (optionnel)
  - CDN pour artifacts statiques
- [ ] Database optimization
  - Index sur colonnes fréquentes
  - Vacuum/Analyze réguliers
  - Connection pooling

#### CI/CD
- [ ] Pipeline automatisé
  - Tests unitaires
  - Tests d'intégration
  - Linting (pylint, black)
- [ ] Deployment strategy
  - Blue-green deployment
  - Canary releases
  - Rollback automatique
- [ ] Version control
  - Git flow
  - Semantic versioning
  - Changelog

### Cloud Deployment

#### AWS

**Services recommandés**:
- **ECS/EKS** - Containers orchestration
- **RDS PostgreSQL** - Managed database
- **ElastiCache Redis** - Managed Redis
- **S3** - Artifacts storage
- **CloudWatch** - Monitoring & Logs
- **IAM** - Access management

**Architecture**:
```
┌─────────────────────────────────────┐
│           Application               │
│        Load Balancer (ALB)          │
└──────────┬──────────────────────────┘
           │
      ┌────┴────┐
      │  ECS    │
      │ Cluster │
      └────┬────┘
           │
   ┌───────┼───────┐
   │       │       │
┌──▼──┐ ┌─▼──┐ ┌──▼──┐
│ Air │ │MLf │ │ Str │
│flow │ │low │ │eaml │
└──┬──┘ └─┬──┘ └──┬──┘
   │      │      │
   └──────┼──────┘
          │
    ┌─────┴─────┐
    │    RDS    │
    │ PostgreSQL│
    └───────────┘
```

**Terraform Example**:
```hcl
# main.tf
provider "aws" {
  region = "us-east-1"
}

# ECS Cluster
resource "aws_ecs_cluster" "mlops" {
  name = "mlops-cluster"
}

# RDS PostgreSQL
resource "aws_db_instance" "airflow" {
  identifier        = "airflow-db"
  engine            = "postgres"
  engine_version    = "13"
  instance_class    = "db.t3.medium"
  allocated_storage = 100
  
  username = var.db_username
  password = var.db_password
  
  multi_az               = true
  backup_retention_period = 7
  skip_final_snapshot    = false
}

# S3 Bucket for MLflow artifacts
resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "mlops-mlflow-artifacts"
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    enabled = true
    
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
  }
}

# ElastiCache Redis
resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "mlops-redis"
  engine               = "redis"
  node_type            = "cache.t3.medium"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
}
```

#### GCP

**Services recommandés**:
- **GKE** - Kubernetes Engine
- **Cloud SQL** - Managed PostgreSQL
- **Cloud Storage** - Artifacts storage
- **Cloud Run** - Serverless containers
- **Stackdriver** - Monitoring

**GKE Deployment**:
```yaml
# kubernetes/airflow-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: airflow-webserver
spec:
  replicas: 2
  selector:
    matchLabels:
      app: airflow-webserver
  template:
    metadata:
      labels:
        app: airflow-webserver
    spec:
      containers:
      - name: webserver
        image: gcr.io/my-project/airflow:latest
        ports:
        - containerPort: 8080
        env:
        - name: AIRFLOW__CORE__EXECUTOR
          value: "KubernetesExecutor"
        - name: AIRFLOW__DATABASE__SQL_ALCHEMY_CONN
          valueFrom:
            secretKeyRef:
              name: airflow-secrets
              key: sql_alchemy_conn
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: airflow-webserver
spec:
  type: LoadBalancer
  selector:
    app: airflow-webserver
  ports:
  - port: 80
    targetPort: 8080
```

#### Azure

**Services recommandés**:
- **AKS** - Azure Kubernetes Service
- **Azure Database for PostgreSQL**
- **Blob Storage** - Artifacts
- **Azure Monitor** - Monitoring
- **Azure DevOps** - CI/CD

### Kubernetes (Production)

**Helm Chart Structure**:
```
mlops-chart/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── airflow-webserver.yaml
│   ├── airflow-scheduler.yaml
│   ├── airflow-worker.yaml
│   ├── mlflow-server.yaml
│   ├── postgres.yaml
│   ├── redis.yaml
│   ├── secrets.yaml
│   └── ingress.yaml
```

**values.yaml**:
```yaml
# values.yaml
airflow:
  image:
    repository: my-registry/airflow
    tag: 2.9.3
  replicas:
    webserver: 2
    scheduler: 2
    worker: 3
  resources:
    webserver:
      requests:
        memory: "2Gi"
        cpu: "1000m"
      limits:
        memory: "4Gi"
        cpu: "2000m"
  
mlflow:
  image:
    repository: my-registry/mlflow
    tag: 2.9.0
  replicas: 2
  storage:
    class: "gp3"
    size: "100Gi"

postgres:
  enabled: true
  persistence:
    enabled: true
    size: "50Gi"
  replication:
    enabled: true
    slaveReplicas: 2

redis:
  enabled: true
  cluster:
    enabled: true
    nodes: 3
  persistence:
    enabled: true
    size: "10Gi"

ingress:
  enabled: true
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: airflow.example.com
      paths:
        - path: /
          service: airflow-webserver
    - host: mlflow.example.com
      paths:
        - path: /
          service: mlflow-server
  tls:
    - secretName: airflow-tls
      hosts:
        - airflow.example.com
    - secretName: mlflow-tls
      hosts:
        - mlflow.example.com
```

**Deployment**:
```bash
# Ajouter le repo Helm
helm repo add mlops https://charts.mlops.io
helm repo update

# Installer
helm install mlops-stack mlops/mlops-chart \
  --values values.yaml \
  --namespace mlops \
  --create-namespace

# Upgrade
helm upgrade mlops-stack mlops/mlops-chart \
  --values values.yaml \
  --namespace mlops

# Rollback
helm rollback mlops-stack 1 --namespace mlops
```

### Auto-Scaling

#### Horizontal Pod Autoscaler (HPA)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: airflow-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: airflow-worker
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

### Monitoring Production

#### Prometheus + Grafana

**Prometheus Configuration**:
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'airflow'
    static_configs:
      - targets: ['airflow-webserver:8080']
    metrics_path: '/admin/metrics'
  
  - job_name: 'mlflow'
    static_configs:
      - targets: ['mlflow-server:5000']
  
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
  
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

**Grafana Dashboards**:
- Airflow Dashboard (ID: 11357)
- PostgreSQL Dashboard (ID: 9628)
- Redis Dashboard (ID: 763)
- Custom MLOps Dashboard

**Alerting Rules**:
```yaml
# alerts.yml
groups:
  - name: mlops_alerts
    interval: 30s
    rules:
      - alert: AirflowTaskFailed
        expr: airflow_task_failed_count > 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Airflow task failed"
          description: "Task {{ $labels.task_id }} in DAG {{ $labels.dag_id }} has failed"
      
      - alert: HighModelRMSE
        expr: mlflow_metric_test_rmse > 1.0
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Model RMSE too high"
          description: "RMSE is {{ $value }}, expected < 1.0"
      
      - alert: DatabaseConnectionsHigh
        expr: pg_stat_database_numbackends > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "PostgreSQL connections high"
          description: "{{ $value }} connections active"
```

### Backup & Disaster Recovery

#### Automated Backups

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)

# PostgreSQL Airflow
docker-compose -f docker-compose-full.yml exec -T postgres-airflow \
  pg_dump -U airflow airflow | gzip > backups/airflow_${DATE}.sql.gz

# PostgreSQL MLflow
docker-compose -f docker-compose-full.yml exec -T postgres-mlflow \
  pg_dump -U mlflow mlflow | gzip > backups/mlflow_${DATE}.sql.gz

# MLflow artifacts
tar -czf backups/mlflow_artifacts_${DATE}.tar.gz mlflow/mlruns/

# Models
tar -czf backups/models_${DATE}.tar.gz models/

# Upload to S3
aws s3 cp backups/ s3://my-mlops-backups/${DATE}/ --recursive

# Retention: Garder 30 derniers jours
find backups/ -name "*.gz" -mtime +30 -delete

echo "✅ Backup completed: ${DATE}"
```

**Cron Job**:
```bash
# crontab -e
0 2 * * * /path/to/backup.sh >> /var/log/mlops-backup.log 2>&1
```

#### Disaster Recovery Plan

1. **RTO (Recovery Time Objective)**: 1 heure
2. **RPO (Recovery Point Objective)**: 24 heures

**Steps**:
```bash
# 1. Restaurer PostgreSQL
gunzip < backups/airflow_20260214.sql.gz | \
  docker-compose -f docker-compose-full.yml exec -T postgres-airflow \
  psql -U airflow airflow

gunzip < backups/mlflow_20260214.sql.gz | \
  docker-compose -f docker-compose-full.yml exec -T postgres-mlflow \
  psql -U mlflow mlflow

# 2. Restaurer artifacts
tar -xzf backups/mlflow_artifacts_20260214.tar.gz -C mlflow/

# 3. Restaurer models
tar -xzf backups/models_20260214.tar.gz -C ./

# 4. Redémarrer les services
docker-compose -f docker-compose-full.yml restart

# 5. Vérifier la santé
./health-check.sh
```

---

## 📚 Références

### Documentation Officielle

- [Apache Airflow](https://airflow.apache.org/docs/apache-airflow/stable/)
- [MLflow](https://mlflow.org/docs/latest/index.html)
- [Scikit-learn](https://scikit-learn.org/stable/documentation.html)
- [Docker Compose](https://docs.docker.com/compose/)
- [PostgreSQL](https://www.postgresql.org/docs/13/index.html)
- [Redis](https://redis.io/documentation)

### Tutoriels et Guides

- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Random Forest Guide](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Docker Multi-Stage Builds](https://docs.docker.com/build/building/multi-stage/)

### Community et Support

- **Airflow Slack**: https://apache-airflow.slack.com
- **MLflow GitHub**: https://github.com/mlflow/mlflow
- **Stack Overflow**: Tags `apache-airflow`, `mlflow`, `scikit-learn`
- **Reddit**: r/datascience, r/MachineLearning

### Outils et Extensions

- **Great Expectations** - Data validation
- **DVC** - Data version control
- **Evidently AI** - Model monitoring
- **WhyLabs** - Data & model observability
- **Seldon Core** - Model serving
- **BentoML** - Model packaging

---

## 📝 Notes de Version

### v2.0.0 (2026-02-14) - MLOps Production Ready

**✨ Features Majeures**:
- ✅ Pipeline MLOps complet end-to-end (7 tâches Airflow)
- ✅ Orchestration Airflow avec CeleryExecutor
- ✅ Tracking MLflow avec Model Registry
- ✅ RandomForest avec 27 features engineered
- ✅ Promotion automatique des modèles (Staging → Production)
- ✅ Reporting JSON automatisé avec numpy type handling
- ✅ Containerisation Docker complète (9 services)
- ✅ Health checks et auto-healing
- ✅ DNS resolution workaround pour MLflow connectivity

**🔧 Modifications Techniques**:
- CarPricePipeline refactorisé avec `base_dir` parameter
- Tous les paths en absolu (`/opt/airflow/project/`)
- XCom data flow avec None checks systématiques
- Conversion numpy types → Python natifs pour JSON serialization
- MLflow API corrections (`sk_model`, `artifact_path`)

**📊 Performances**:
- **Dataset**: 24,776 lignes → 7,524 après nettoyage
- **Features**: 27 (3 numériques + 9 catégorielles + 15 binaires)
- **Train/Test**: 6,019 / 1,505 samples
- **Métriques**:
  - R² Train: 0.87 (86.89%)
  - R² Test: 0.73 (72.99%)
  - RMSE: 0.52
  - MAE: 0.39

**🐛 Bugs Fixés**:
- MLflow "Invalid Host header" → DNS IP resolution
- FileNotFoundError → Absolute paths
- TypeError numpy bool_ → Type conversion
- XCom None handling → Systematic checks
- Version string serialization → Explicit str()

**🏗️ Infrastructure**:
- 9 services Docker en production
- 2 bases PostgreSQL (Airflow + MLflow)
- Redis message broker
- Custom Airflow image avec ML stack
- Network bridge avec DNS interne

**📚 Documentation**:
- README.md mis à jour (structure complète)
- DOCUMENTATION_COMPLETE.md créé (ce fichier)
- Troubleshooting exhaustif
- Guides de déploiement Production

### v1.1.0 (2026-02-01)

- Ajout Docker Compose
- Integration MLflow
- Application Streamlit

### v1.0.0 (2025-12-01)

- Version initiale
- Modèle RandomForest de base
- Pipeline de données

---

## 🤝 Contribution

### Guidelines

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

### Standards de Code

```bash
# Formatting
black scripts/ airflow/dags/ --line-length 100

# Linting
pylint scripts/ airflow/dags/ --rcfile=.pylintrc

# Type checking
mypy scripts/ airflow/dags/ --strict

# Tests
pytest tests/ -v --cov=. --cov-report=html
```

### Commit Convention

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: Nouvelle fonctionnalité
- `fix`: Correction de bug
- `docs`: Documentation seulement
- `style`: Formatage (pas de changement de code)
- `refactor`: Refactoring
- `test`: Ajout de tests
- `chore`: Maintenance

**Exemple**:
```
feat(pipeline): add data quality validation task

- Add check_data_quality function
- Validate row count and missing percentage
- Push results to XCom for downstream tasks

Closes #123
```

---

## 📄 License

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

## 👥 Auteurs et Contacts

- **Data Science Team** - alerts@example.com
- **DevOps Team** - devops@example.com
- **GitHub** - https://github.com/Azaziop/detection_car_price

---

## 🙏 Remerciements

- Apache Airflow Community pour l'orchestrateur robuste
- MLflow Team pour le tracking framework
- Scikit-learn Contributors pour les outils ML
- Docker Team pour la containerisation
- PostgreSQL Team pour la base de données
- Redis Team pour le message broker

---

**Date de dernière mise à jour**: 14 février 2026  
**Version de la documentation**: 2.0.0  
**Statut**: ✅ Production Ready
