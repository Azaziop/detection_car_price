# 🐳 Stack MLOps Dockerisée - Guide Complet

## Vue d'ensemble

Cette stack complète intègre tous les services MLOps dans Docker:
- **MLflow** : Tracking d'expériences et registry de modèles
- **Streamlit** : Interface web de prédiction
- **Airflow** : Orchestration des pipelines ML
- **PostgreSQL** : Base de données pour Airflow et MLflow
- **Redis** : Message broker pour Airflow

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Docker Compose Stack                │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌──────────────┐  ┌──────────────┐            │
│  │  PostgreSQL  │  │    Redis     │            │
│  │   :5432      │  │   :6379      │            │
│  └──────┬───────┘  └──────┬───────┘            │
│         │                 │                     │
│  ┌──────▼──────────────────▼───────┐           │
│  │     MLflow Tracking Server      │           │
│  │     http://localhost:5000       │           │
│  │  Backend: PostgreSQL (mlflow DB)│           │
│  └──────┬──────────────────────────┘           │
│         │                                       │
│  ┌──────▼──────────────────────────┐           │
│  │     Streamlit Application       │           │
│  │     http://localhost:8501       │           │
│  │  Uses: MLflow for model loading │           │
│  └─────────────────────────────────┘           │
│                                                  │
│  ┌─────────────────────────────────┐           │
│  │      Airflow Stack              │           │
│  │  ┌──────────────────────────┐   │           │
│  │  │  Webserver :8080         │   │           │
│  │  │  (Interface Web)         │   │           │
│  │  └──────────────────────────┘   │           │
│  │  ┌──────────────────────────┐   │           │
│  │  │  Scheduler               │   │           │
│  │  │  (Planification DAGs)    │   │           │
│  │  └──────────────────────────┘   │           │
│  │  ┌──────────────────────────┐   │           │
│  │  │  Worker (Celery)         │   │           │
│  │  │  (Exécution des tâches)  │   │           │
│  │  └──────────────────────────┘   │           │
│  │  ┌──────────────────────────┐   │           │
│  │  │  Triggerer               │   │           │
│  │  │  (Gestion événements)    │   │           │
│  │  └──────────────────────────┘   │           │
│  └─────────────────────────────────┘           │
└─────────────────────────────────────────────────┘
```

## 🚀 Démarrage Rapide

### Première Utilisation

```bash
# Donner les droits d'exécution aux scripts
chmod +x docker-start-full.sh docker-stop-full.sh docker-reset-full.sh

# Démarrer toute la stack
./docker-start-full.sh
```

Le script va:
1. Arrêter les services locaux existants
2. Construire les images Docker
3. Démarrer tous les services
4. Créer les utilisateurs Airflow
5. Afficher l'état final

**Temps de premier démarrage:** ~3-5 minutes (construction des images)

### Utilisation Quotidienne

```bash
# Démarrer la stack
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
docker compose -f docker-compose-full.yml up -d

# Arrêter la stack
docker compose -f docker-compose-full.yml down

# Ou utiliser les scripts
./docker-start-full.sh
./docker-stop-full.sh
```

## 📊 Accès aux Services

| Service | URL | Identifiants |
|---------|-----|--------------|
| **MLflow** | http://localhost:5000 | Aucun (accès direct) |
| **Streamlit** | http://localhost:8501 | Aucun (accès direct) |
| **Airflow** | http://localhost:8080 | `admin` / `admin` |
| **PostgreSQL** | localhost:5432 | `airflow` / `airflow` |

### Connexion Airflow

Si vous rencontrez "Identifiants invalides":
```bash
# Réinitialiser complètement Airflow
./docker-reset-full.sh
```

Les identifiants par défaut sont:
- **Username:** `admin`
- **Password:** `admin`

## 🔧 Commandes Docker Utiles

### Gestion des Services

```bash
# Démarrer tous les services
docker compose -f docker-compose-full.yml up -d

# Arrêter tous les services
docker compose -f docker-compose-full.yml down

# Redémarrer un service spécifique
docker compose -f docker-compose-full.yml restart mlflow
docker compose -f docker-compose-full.yml restart streamlit
docker compose -f docker-compose-full.yml restart airflow-webserver

# Voir l'état des services
docker compose -f docker-compose-full.yml ps
```

### Logs et Debugging

```bash
# Voir tous les logs en temps réel
docker compose -f docker-compose-full.yml logs -f

# Logs d'un service spécifique
docker compose -f docker-compose-full.yml logs -f mlflow
docker compose -f docker-compose-full.yml logs -f streamlit
docker compose -f docker-compose-full.yml logs -f airflow-webserver
docker compose -f docker-compose-full.yml logs -f airflow-scheduler

# Dernières 100 lignes de logs
docker compose -f docker-compose-full.yml logs --tail=100 airflow-webserver
```

### Accès aux Containers

```bash
# Exécuter une commande dans un container
docker compose -f docker-compose-full.yml exec mlflow bash
docker compose -f docker-compose-full.yml exec streamlit bash
docker compose -f docker-compose-full.yml exec airflow-webserver bash

# Lister les utilisateurs Airflow
docker compose -f docker-compose-full.yml exec airflow-webserver airflow users list

# Créer un nouvel utilisateur Airflow
docker compose -f docker-compose-full.yml exec airflow-webserver \
  airflow users create \
  --username newuser \
  --firstname New \
  --lastname User \
  --role Admin \
  --email new@example.com \
  --password newpassword
```

### Nettoyage

```bash
# Arrêter et supprimer les containers
docker compose -f docker-compose-full.yml down

# Supprimer aussi les volumes (ATTENTION: perte de données)
docker compose -f docker-compose-full.yml down -v

# Supprimer les images construites
docker compose -f docker-compose-full.yml down --rmi all
```

## 📦 Volumes et Données Persistantes

Les données sont sauvegardées dans les volumes suivants:

| Volume/Répertoire | Contenu |
|-------------------|---------|
| `postgres-db-volume` | Base de données PostgreSQL (Airflow + MLflow) |
| `./mlflow/mlruns` | Expériences et artifacts MLflow |
| `./models` | Modèles ML entraînés |
| `./data` | Dataset et données |
| `./airflow/logs` | Logs Airflow |
| `./airflow/dags` | DAGs Airflow |

**Important:** Ne supprimez pas ces répertoires si vous voulez conserver vos données.

## 🔄 Workflow MLOps

### 1. Entraîner un Modèle

```bash
# Exécuter l'entraînement dans le container Streamlit
docker compose -f docker-compose-full.yml exec streamlit python scripts/train_with_mlflow.py
```

Le modèle sera:
- Enregistré dans MLflow (visible sur http://localhost:5000)
- Sauvegardé localement dans `./models/`
- Utilisable immédiatement dans Streamlit

### 2. Faire des Prédictions

1. Ouvrir http://localhost:8501
2. Entrer les caractéristiques du véhicule
3. Cliquer sur "Prédire"

L'application charge automatiquement le modèle depuis MLflow.

### 3. Orchestrer avec Airflow

1. Ouvrir http://localhost:8080
2. Se connecter avec `admin` / `admin`
3. Activer le DAG `car_price_predictor_pipeline`
4. Déclencher manuellement ou attendre l'exécution planifiée

Le pipeline va:
- Vérifier la qualité des données
- Entraîner le modèle
- L'évaluer
- Le promouvoir en staging
- Le valider
- Le déployer en production

### 4. Suivre les Expériences

1. Ouvrir http://localhost:5000
2. Explorer les expériences
3. Comparer les modèles
4. Voir les métriques et artifacts

## 🛠️ Résolution de Problèmes

### Problème: Identifiants Airflow invalides

**Solution:**
```bash
./docker-reset-full.sh
```

Cela va recréer complètement la base de données et les utilisateurs.

### Problème: Port déjà utilisé

**Erreur:** `Bind for 0.0.0.0:8080 failed: port is already allocated`

**Solution:**
```bash
# Voir qui utilise le port
lsof -i :8080
lsof -i :5000
lsof -i :8501

# Arrêter le processus ou modifier les ports dans docker-compose-full.yml
```

### Problème: Services ne démarrent pas

**Solution:**
```bash
# Voir les logs pour identifier l'erreur
docker compose -f docker-compose-full.yml logs

# Reconstruire les images
docker compose -f docker-compose-full.yml build --no-cache

# Redémarrer
docker compose -f docker-compose-full.yml up -d
```

### Problème: Espace disque insuffisant

**Solution:**
```bash
# Nettoyer les images et containers inutilisés
docker system prune -a

# Voir l'utilisation du disque
docker system df
```

### Problème: MLflow ne trouve pas les modèles

**Vérifier la configuration:**
```bash
# Dans le container Streamlit
docker compose -f docker-compose-full.yml exec streamlit env | grep MLFLOW
```

Devrait afficher: `MLFLOW_TRACKING_URI=http://mlflow:5000`

## 🎯 Avantages de cette Architecture

### ✅ Isolation Complète
- Chaque service dans son propre container
- Pas de conflits de dépendances
- Environnement reproductible

### ✅ Scalabilité
- Facile d'ajouter des workers Airflow
- Load balancing possible
- Déploiement multi-instance

### ✅ Production-Ready
- Architecture professionnelle
- Monitoring intégré
- Logs centralisés

### ✅ Portabilité
- Fonctionne sur Linux, macOS, Windows
- Même comportement partout
- Déploiement cloud simple

### ✅ Maintenance Simplifiée
- Mises à jour faciles
- Rollback rapide
- Backup automatisé

## 🚀 Passage en Production

### 1. Sécurité

```bash
# Changer les mots de passe
# Éditer docker-compose-full.yml:
POSTGRES_PASSWORD: your_secure_password
_AIRFLOW_WWW_USER_PASSWORD: your_secure_admin_password
```

### 2. Reverse Proxy (HTTPS)

Ajouter nginx pour HTTPS:
```yaml
nginx:
  image: nginx:alpine
  ports:
    - "443:443"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf
    - ./ssl:/etc/nginx/ssl
```

### 3. Monitoring

Ajouter Prometheus + Grafana:
```bash
# Voir la documentation de monitoring
# (à créer séparément)
```

### 4. Backup Automatique

```bash
# Script de backup PostgreSQL
docker compose -f docker-compose-full.yml exec postgres \
  pg_dump -U airflow airflow > backup_airflow_$(date +%Y%m%d).sql

docker compose -f docker-compose-full.yml exec postgres \
  pg_dump -U airflow mlflow > backup_mlflow_$(date +%Y%m%d).sql
```

## 📚 Ressources

- **Docker Compose**: https://docs.docker.com/compose/
- **MLflow**: https://mlflow.org/docs/latest/
- **Airflow**: https://airflow.apache.org/docs/
- **Streamlit**: https://docs.streamlit.io/

## 🎓 Tutoriel Complet

### Scenario: Premier Déploiement

```bash
# 1. Cloner le repository
git clone https://github.com/Azaziop/detection_car_price.git
cd detection_car_price

# 2. Donner les droits d'exécution
chmod +x docker-start-full.sh docker-stop-full.sh docker-reset-full.sh

# 3. Démarrer la stack
./docker-start-full.sh
# Attendre 3-5 minutes...

# 4. Vérifier que tout fonctionne
docker compose -f docker-compose-full.yml ps

# 5. Tester MLflow
open http://localhost:5000

# 6. Tester Streamlit
open http://localhost:8501

# 7. Tester Airflow
open http://localhost:8080
# Login: admin / admin

# 8. Entraîner un modèle
docker compose -f docker-compose-full.yml exec streamlit \
  python scripts/train_with_mlflow.py

# 9. Voir les résultats dans MLflow
# Rafraîchir http://localhost:5000

# 10. Activer le pipeline Airflow
# Dans l'interface Airflow, activer le DAG car_price_predictor_pipeline
```

Félicitations! Votre stack MLOps est opérationnelle! 🎉
