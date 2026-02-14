# 🚀 Guide de démarrage Apache Airflow

## Installation et Configuration

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Initialiser Airflow
```bash
chmod +x start_airflow.sh
./start_airflow.sh
```

### 3. Démarrer Airflow

**Terminal 1 - Webserver:**
```bash
export AIRFLOW_HOME=$(pwd)/airflow
airflow webserver --port 8080
```

**Terminal 2 - Scheduler:**
```bash
export AIRFLOW_HOME=$(pwd)/airflow
airflow scheduler
```

### 4. Accéder à l'interface web
- URL: http://localhost:8080
- Username: `admin`
- Password: `admin`

---

## 📋 Structure du Pipeline

Le DAG `car_price_predictor_pipeline` orchestre les étapes suivantes:

```
1. check_data_quality      → Vérification de la qualité des données
   ↓
2. train_model             → Entraînement du modèle avec MLflow
   ↓
3. evaluate_model          → Évaluation des performances (R², RMSE)
   ↓
4. promote_to_staging      → Promotion vers Staging si critères OK
   ↓
5. validate_staging_model  → Tests de validation en Staging
   ↓
6. promote_to_production   → Promotion vers Production
   ↓
7. send_pipeline_report    → Génération du rapport
```

---

## 🎯 Fonctionnalités du Pipeline

### Vérification de la Qualité des Données
- Compte les lignes et colonnes
- Calcule le pourcentage de valeurs manquantes
- Bloque l'entraînement si les données sont insuffisantes

### Entraînement du Modèle
- Utilise `CarPricePipeline` existant
- Track avec MLflow (métriques, paramètres, artifacts)
- Tag automatique avec "pipeline: airflow"

### Évaluation
- Critères de qualité:
  - R² minimum: 0.80
  - RMSE maximum: 50,000
- Décide automatiquement de la promotion

### Promotion Multi-Stage
- **Staging**: Environnement de test
- **Validation**: Tests avant production
- **Production**: Déploiement automatique si validé

### Reporting
- Génère un rapport JSON complet
- Sauvegardé dans `reports/pipeline_report_*.json`

---

## 📅 Configuration du Planning

Le DAG s'exécute **automatiquement chaque semaine** (`@weekly`).

Pour modifier le planning, éditez `car_price_ml_pipeline.py`:
```python
schedule_interval='@daily'    # Quotidien
schedule_interval='@weekly'   # Hebdomadaire (actuel)
schedule_interval='0 2 * * 1' # Tous les lundis à 2h du matin
```

---

## 🛠️ Commandes Utiles

### Liste des DAGs
```bash
export AIRFLOW_HOME=$(pwd)/airflow
airflow dags list
```

### Tester le DAG manuellement
```bash
airflow dags test car_price_predictor_pipeline $(date +%Y-%m-%d)
```

### Activer/Désactiver le DAG
```bash
# Activer
airflow dags unpause car_price_predictor_pipeline

# Désactiver
airflow dags pause car_price_predictor_pipeline
```

### Exécuter une tâche spécifique
```bash
airflow tasks test car_price_predictor_pipeline train_model $(date +%Y-%m-%d)
```

### Voir les logs d'une tâche
```bash
airflow tasks logs car_price_predictor_pipeline train_model $(date +%Y-%m-%d)
```

### Déclencher manuellement
```bash
airflow dags trigger car_price_predictor_pipeline
```

---

## 📊 Surveillance et Monitoring

### Interface Web
1. **DAGs** - Vue d'ensemble de tous les pipelines
2. **Grid** - Historique des exécutions
3. **Graph** - Visualisation du flux de tâches
4. **Gantt** - Durée d'exécution
5. **Code** - Code source du DAG

### Indicateurs de Statut
- 🟢 **Success**: Tâche réussie
- 🔴 **Failed**: Tâche échouée
- 🟡 **Running**: En cours d'exécution
- ⚪ **Queued**: En attente
- 🔵 **Skipped**: Ignorée

---

## 🔧 Configuration Avancée

### Modifier les critères de qualité
Éditez `airflow/dags/car_price_ml_pipeline.py`:
```python
MINIMUM_R2 = 0.85      # Augmenter l'exigence
MAXIMUM_RMSE = 40000   # Réduire l'erreur acceptable
```

### Ajouter des notifications email
Dans `default_args`:
```python
'email': ['votre-email@example.com'],
'email_on_failure': True,
'email_on_success': True,
```

### Parallélisation des tâches
Changer l'executor dans `airflow/config/airflow.cfg`:
```ini
executor = LocalExecutor  # Plus rapide que SequentialExecutor
```

---

## 🐛 Dépannage

### DAG non détecté
```bash
# Vérifier les erreurs de syntaxe
python airflow/dags/car_price_ml_pipeline.py

# Forcer le rafraîchissement
airflow dags list-import-errors
```

### Erreur d'import
Vérifiez que le PYTHONPATH inclut le projet:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Base de données bloquée
```bash
# Réinitialiser la base de données
rm airflow/airflow.db
airflow db init
```

---

## 📈 Intégration avec MLflow

Le pipeline est totalement intégré avec MLflow:
- Tous les runs sont trackés
- Modèles automatiquement enregistrés
- Promotion via Model Registry
- Métriques et artifacts sauvegardés

Voir les résultats dans MLflow UI:
```bash
mlflow ui --backend-store-uri file:./mlflow/mlruns
```

---

## 🎓 Prochaines Étapes

1. ✅ Initialiser Airflow avec `./start_airflow.sh`
2. ✅ Accéder à http://localhost:8080
3. ✅ Activer le DAG `car_price_predictor_pipeline`
4. ✅ Observer la première exécution
5. ✅ Consulter les rapports dans `reports/`

Pour toute question, consultez:
- [Documentation Airflow](https://airflow.apache.org/docs/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
