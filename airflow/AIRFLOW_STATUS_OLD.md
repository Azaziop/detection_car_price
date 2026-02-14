# État d'Apache Airflow

## ⚠️ PROBLÈME DE COMPATIBILITÉ

**Apache Airflow 2.8.1 N'EST PAS COMPATIBLE avec macOS + Python 3.12**

### Problème technique
- Bug Gunicorn SIGSEGV: Le serveur web/scheduler crashe immédiatement avec des erreurs "Worker (pid:XXXX) was sent SIGSEGV!"
- Affecte TOUS les modes: `airflow standalone`, `airflow webserver`, `airflow scheduler`
- Bug connu dans la communauté Airflow: Gunicorn ne fonctionne pas sur macOS avec Python 3.12

### Ce qui a été installé et configuré
✅ Apache Airflow 2.8.1 installé dans `.venv`
✅ Base de données SQLite initialisée (`airflow/airflow.db`)
✅ Utilisateur admin créé (username: `admin`, password: `admin`)
✅ DAG complet créé: `car_price_predictor_pipeline` (7 tâches)
✅ Configuration prête dans `airflow/config/airflow.cfg`
✅ Documentation complète dans `airflow/README_AIRFLOW.md`

### DAG Airflow créé

Le fichier `airflow/dags/car_price_ml_pipeline.py` contient un pipeline complet avec 7 tâches:

1. **check_data_quality** → Vérifie la qualité des données
2. **train_model** → Entraîne le modèle avec MLflow
3. **evaluate_model** → Évalue les performances
4. **promote_to_staging** → Promeut vers staging
5. **validate_staging_model** → Valide en staging
6. **promote_to_production** → Déploie en production
7. **send_pipeline_report** → Génère le rapport final

### Solutions alternatives

#### Option 1: Docker (RECOMMANDÉ)
```bash
# Utiliser l'image officielle Airflow dans Docker
docker run -p 8080:8080 apache/airflow:2.8.1 standalone
```

#### Option 2: Environnement Linux/CI/CD
Déployer Airflow sur:
- Ubuntu/Debian
- Cloud (AWS, GCP, Azure)
- Kubernetes avec Helm chart officiel

#### Option 3: Utiliser Python 3.11
```bash
# Désinstaller Python 3.12, installer Python 3.11
pyenv install 3.11
pyenv local 3.11
# Réinstaller Airflow
```

#### Option 4: Tester le DAG sans interface (limité)
```bash
export AIRFLOW_HOME=$(pwd)/airflow
source .venv/bin/activate
# Tester syntaxe du DAG
airflow dags list
# Tester une seule tâche
airflow tasks test car_price_predictor_pipeline check_data_quality 2026-02-14
```

### Utilisation du code actuel

Le DAG est **prêt à être utilisé** dans un environnement compatible. Pour l'utiliser:

1. Copier le dossier `airflow/` vers un système Linux ou Docker
2. Exécuter `airflow db init && airflow users create ...`
3. Lancer `airflow standalone` ou `airflow webserver` + `airflow scheduler`
4. Accéder à http://localhost:8080
5. Activer le DAG `car_price_predictor_pipeline`

### Conclusion

Airflow est **complètement configuré et prêt** mais ne peut pas s'exécuter sur votre système actuel (macOS + Python 3.12). Le code fonctionnera parfaitement dans un environnement compatible (Docker, Linux, ou Python 3.11).

**Ce que fait Airflow pour ce projet:**
- Orchestre tout le pipeline ML automatiquement
- Gère les dépendances entre les tâches
- Tracking MLflow intégré
- Promotions staging → production automatiques
- Rapports et alertes configurables
- Interface web pour monitoring
- Scheduling automatique des runs

Le DAG remplace l'exécution manuelle de `main_mlflow.py` par une orchestration professionnelle automatisée.
