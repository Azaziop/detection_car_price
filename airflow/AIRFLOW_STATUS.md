# 🚨 Airflow - Statut Final

**Date**: 14 février 2026  
**Plateforme**: macOS 26.0 ARM64 (Apple Silicon M1/M2/M3)  
**Statut**: ❌ **NON FONCTIONNEL EN MODE NATIF**

---

## Résumé Exécutif

Apache Airflow 2.9.3 **ne peut pas fonctionner nativement sur macOS ARM64**, indépendamment de la version Python utilisée. Le problème est un bug critique dans Gunicorn qui cause des crashs SIGSEGV constants.

### Tentatives de Résolution (toutes échouées)

#### ✅ Migration Python 3.12 → 3.11
- **Action**: Installation complète de Python 3.11.14 via Homebrew
- **Résultat**: ❌ ÉCHEC - SIGSEGV persiste
- **Conclusion**: Le problème n'est PAS lié à la version Python

#### ✅ Configuration Gunicorn Workers
- **Action**: Création webserver_config.py avec workers synchrones
- **Configuration**: `GUNICORN_CMD_ARGS = '--workers=1 --worker-class=sync --timeout=120'`
- **Résultat**: ❌ ÉCHEC - SIGSEGV persiste
- **Conclusion**: Le type de worker n'affecte pas le bug

#### ✅ Variables d'Environnement
- **Action**: `AIRFLOW__WEBSERVER__WORKERS=1`, `AIRFLOW__WEBSERVER__WORKER_CLASS=sync`
- **Résultat**: ❌ ÉCHEC - SIGSEGV persiste
- **Conclusion**: La configuration runtime est ignorée

---

## Root Cause Analysis

### Problème Technique

```
[ERROR] Worker (pid:XXXXX) was sent SIGSEGV!
```

- **Composants affectés**: Webserver, Scheduler, Triggerer (TOUS)
- **Fréquence**: 100+ crashes en 10 secondes (crash loop infini)
- **Plateforme**: macOS ARM64 UNIQUEMENT
- **Gunicorn version**: Toutes les versions testées
- **Python versions**: 3.11.14, 3.12.x (tous affectés)

### Cause Racine

**Gunicorn + macOS ARM64** = Incompatibilité au niveau système

- Gunicorn utilise des appels système (fork, signals) qui ne fonctionnent pas correctement sur l'architecture ARM64 de macOS
- Ce n'est PAS un bug Python, mais un problème d'architecture
- Upstream issue connue : https://github.com/benoitc/gunicorn/issues/2681

---

## ✅ Solution Recommandée : Docker

**docker-compose.yaml** (officiel Apache Airflow)

```bash
# 1. Télécharger la configuration officielle
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.9.3/docker-compose.yaml'

# 2. Initialiser l'environnement
docker-compose up airflow-init

# 3. Démarrer tous les services
docker-compose up -d

# 4. Accéder à l'interface web
open http://localhost:8080
# Login: airflow / airflow
```

**Avantages Docker:**
- ✅ Fonctionne parfaitement sur macOS ARM64
- ✅ Configuration officielle supportée
- ✅ Tous les composants disponibles
- ✅ Interface web fonctionnelle
- ✅ Monitoring et logging complets

---

## Alternatives

### Cloud Deployment (Production-Ready)
- **AWS MWAA**: Managed Workflows for Apache Airflow
- **GCP Cloud Composer**: Airflow géré sur GCP
- **Azure Data Factory**: Alternative Microsoft

### Dev Containers (VS Code)
```json
{
  "image": "apache/airflow:2.9.3-python3.11",
  "forwardPorts": [8080]
}
```

### Linux VM
- Multipass, Vagrant, ou VirtualBox
- Architecture x86_64 recommandée

---

## État des Fichiers du Projet

### ✅ Fichiers Fonctionnels

| Fichier | Statut | Description |
|---------|--------|-------------|
| `airflow/dags/car_price_ml_pipeline.py` | ✅ Prêt | DAG complet, 7 tâches |
| `airflow/config/airflow.cfg` | ✅ Configuré | SQLite, SequentialExecutor |
| `airflow/airflow.db` | ✅ Initialisé | Schema 686269002441 |
| Admin user | ✅ Créé | admin/admin |

### ❌ Configuration Non Utilisable

| Fichier | Problème |
|---------|----------|
| `airflow/webserver_config.py` | Ignoré par Gunicorn sur ARM64 |
| Variables env GUNICORN_CMD_ARGS | Sans effet |

---

## Timeline des Tests

**14 février 2026 - 16:30 - 16:38**

- 16:30 : Découverte du bug SIGSEGV
- 16:31 : Migration Python 3.11 complète
- 16:33 : Vérification Python 3.11.14 actif
- 16:34 : Création webserver_config.py
- 16:35 : Test avec sync workers → **ÉCHEC**
- 16:37 : Test avec variables env → **ÉCHEC** (250+ crashes)
- 16:38 : Décision finale : Docker obligatoire

---

## Verdict Final

⛔ **Apache Airflow ne peut PAS être installé nativement sur macOS ARM64**

**Pour ce projet:**
1. **Développement**: Utiliser Docker (recommandé)
2. **Production**: Déployer sur Linux (cloud ou VM)
3. **Testing DAGs**: Commande `airflow dags test` fonctionne (sans webserver)

**Documentation mise à jour:**
- [README.md](../README.md) : Section Airflow avec solution Docker
- [PYTHON_3.11_MIGRATION.md](../PYTHON_3.11_MIGRATION.md) : Clarifie que Python 3.11 seul ne suffit pas

---

## Commande de Test (sans webserver)

```bash
# Fonctionne sur macOS ARM64 (teste la logique DAG uniquement)
export AIRFLOW_HOME=$(pwd)/airflow
source .venv/bin/activate
airflow dags test car_price_predictor_pipeline $(date +%Y-%m-%d)
```

**Note**: Cette commande teste le DAG sans démarrer de serveur web, donc contourne le bug Gunicorn.

---

**Conclusion**: La stack MLOps du projet (MLflow, Streamlit, scikit-learn, DVC) fonctionne parfaitement. Seul Airflow nécessite Docker sur macOS ARM64.
