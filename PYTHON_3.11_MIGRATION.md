# Migration vers Python 3.11 - Guide complet

## 🎯 Pourquoi Python 3.11?

Apache Airflow 2.8.1 a un bug de compatibilité avec Python 3.12 sur macOS (erreurs Gunicorn SIGSEGV). **Python 3.11 résout ce problème**.

## 📋 Méthodes d'installation Python 3.11

### Option 1: Homebrew (Recommandé pour macOS)

```bash
# Installer Python 3.11
brew install python@3.11

# Vérifier l'installation
python3.11 --version  # Devrait afficher: Python 3.11.x
```

### Option 2: pyenv (Gestion multiple versions)

```bash
# Installer pyenv si nécessaire
brew install pyenv

# Installer Python 3.11
pyenv install 3.11.7  # Version stable

# Définir comme version locale pour ce projet
cd /Users/anass/PycharmProjects/PythonProject9
pyenv local 3.11.7

# Vérifier
python --version  # Devrait afficher: Python 3.11.7
```

### Option 3: python.org

Télécharger depuis: https://www.python.org/downloads/release/python-3117/

## 🔄 Migration du projet

### Étape 1: Sauvegarder l'environnement actuel

```bash
cd /Users/anass/PycharmProjects/PythonProject9

# Exporter les packages actuels
source .venv/bin/activate
pip freeze > requirements_backup_python312.txt
deactivate
```

### Étape 2: Supprimer l'ancien environnement virtuel

```bash
# Sauvegarder les fichiers importants de venv si nécessaire
rm -rf .venv
```

### Étape 3: Créer un nouvel environnement avec Python 3.11

#### Si vous utilisez pyenv:
```bash
pyenv local 3.11.7
python -m venv .venv
```

#### Si vous utilisez Homebrew:
```bash
python3.11 -m venv .venv
```

#### Si installation depuis python.org:
```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m venv .venv
```

### Étape 4: Activer le nouvel environnement

```bash
source .venv/bin/activate

# Vérifier la version Python dans venv
python --version  # Doit afficher Python 3.11.x
```

### Étape 5: Réinstaller les dépendances

```bash
# Mettre à jour pip
pip install --upgrade pip

# Réinstaller toutes les dépendances
pip install -r requirements.txt

# Si problème, réinstaller depuis le backup
pip install -r requirements_backup_python312.txt

# Pour les dépendances de développement
pip install -r requirements/requirements-dev.txt
```

### Étape 6: Réinitialiser Airflow

```bash
# Supprimer l'ancienne base de données Airflow
rm -rf airflow/airflow.db
rm -rf airflow/logs/*
rm -rf airflow/airflow-webserver.pid

# Définir AIRFLOW_HOME
export AIRFLOW_HOME=$(pwd)/airflow

# Réinitialiser la base de données
airflow db init

# Recréer l'utilisateur admin
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

### Étape 7: Tester Airflow

```bash
# Lister les DAGs (devrait fonctionner sans erreur)
airflow dags list

# Démarrer Airflow standalone
airflow standalone
```

**L'interface devrait maintenant être accessible à:** http://localhost:8080

## ✅ Vérification de la migration

### Checklist de vérification:

```bash
# 1. Version Python correcte
python --version  # Python 3.11.x

# 2. Environnement virtuel activé
which python  # Doit pointer vers .venv/bin/python

# 3. Packages installés
pip list | grep -E "airflow|streamlit|mlflow"

# 4. Airflow fonctionne
airflow dags list  # Pas d'erreur SIGSEGV

# 5. Application Streamlit
streamlit run main_mlflow.py  # Doit démarrer normalement

# 6. Tests passent
pytest tests/ -v  # Tous les tests OK
```

## 🐛 Troubleshooting

### Erreur: "Command not found: python3.11"

**Solution Homebrew:**
```bash
brew install python@3.11
# Ajouter au PATH
echo 'export PATH="/opt/homebrew/opt/python@3.11/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**Solution pyenv:**
```bash
# Installer pyenv
brew install pyenv

# Configurer shell
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc

# Installer Python 3.11
pyenv install 3.11.7
pyenv local 3.11.7
```

### Erreur lors de l'installation de packages

```bash
# Mettre à jour pip, setuptools, wheel
pip install --upgrade pip setuptools wheel

# Installer un package à la fois en cas d'échec
pip install apache-airflow==2.8.1
pip install streamlit
pip install mlflow
# etc.
```

### Airflow lance toujours des erreurs SIGSEGV

```bash
# Vérifier la version Python dans le venv
source .venv/bin/activate
python --version  # DOIT être 3.11.x

# Si c'est encore 3.12, recréer le venv:
deactivate
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Commande `pyenv local` ne fonctionne pas

```bash
# Créer manuellement le fichier .python-version
echo "3.11.7" > .python-version

# Vérifier
cat .python-version
python --version
```

## 📊 Comparaison des performances

### Python 3.12 (Actuel - Non compatible Airflow)
❌ Airflow: Crashe avec erreurs SIGSEGV  
✅ Streamlit: Fonctionne  
✅ MLflow: Fonctionne  
✅ Tests: Passent  

### Python 3.11 (Recommandé)
✅ Airflow: Fonctionne parfaitement  
✅ Streamlit: Fonctionne  
✅ MLflow: Fonctionne  
✅ Tests: Passent  
✅ Performances: Comparables à 3.12  

## 🎯 Commandes rapides (Résumé)

```bash
# Installation Python 3.11
brew install python@3.11

# Migration complète
cd /Users/anass/PycharmProjects/PythonProject9
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Réinitialiser Airflow
rm -rf airflow/airflow.db
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init
airflow users create --username admin --password admin --firstname Admin \
    --lastname User --role Admin --email admin@example.com

# Tester
airflow standalone  # Devrait fonctionner!
```

## 📚 Ressources

- [Python 3.11 Release Notes](https://docs.python.org/3/whatsnew/3.11.html)
- [pyenv Documentation](https://github.com/pyenv/pyenv)
- [Homebrew Python](https://docs.brew.sh/Homebrew-and-Python)
- [Airflow Installation](https://airflow.apache.org/docs/apache-airflow/stable/installation/index.html)

## ⚠️ Notes importantes

1. **Pas de downgrade de Python système**: Ne pas toucher à la version Python système de macOS
2. **Utiliser venv**: Toujours travailler dans un environnement virtuel
3. **Tester avant de supprimer**: Sauvegarder requirements avec `pip freeze`
4. **MLflow runs préservés**: Les données MLflow dans `mlflow/mlruns/` ne sont pas affectées
5. **Models intacts**: Les modèles dans `models/` fonctionneront avec Python 3.11

## 🎉 Après la migration

Une fois migré vers Python 3.11, vous pourrez:

✅ Utiliser Airflow avec interface web complète  
✅ Orchestrer automatiquement votre pipeline ML  
✅ Monitorer les DAGs en temps réel  
✅ Planifier des entraînements réguliers  
✅ Gérer les promotions Staging → Production  

**Temps estimé de migration:** 15-20 minutes

---

**Date:** Février 2026  
**Version du guide:** 1.0
