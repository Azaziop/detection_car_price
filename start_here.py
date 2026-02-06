"""
Guide d'utilisation du projet MLOps
"""

print("""
╔═══════════════════════════════════════════════════════════════╗
║   🚗 CAR PRICE PREDICTION - MLOps Project avec MLflow + DVC  ║
╚═══════════════════════════════════════════════════════════════╝

✅ Installation terminée!

📋 COMMANDES PRINCIPALES:

1️⃣  ENTRAÎNER LE MODÈLE
   python3 train_with_mlflow.py
   
   → Entraîne le modèle avec tracking MLflow
   → Sauvegarde les métriques, hyperparamètres et artifacts
   → Crée les fichiers .pkl nécessaires

2️⃣  VOIR LES EXPÉRIENCES (MLflow UI)
   mlflow ui
   
   → Ouvre l'interface web sur http://localhost:5000
   → Compare les runs, métriques, hyperparamètres
   → Visualise les plots et artifacts

3️⃣  LANCER L'APPLICATION STREAMLIT
   streamlit run main_mlflow.py
   
   → Version avec intégration MLflow
   → Peut charger les modèles depuis le Model Registry
   
   OU (version simple):
   streamlit run main.py

4️⃣  VÉRIFIER LES MODÈLES ENREGISTRÉS
   python3 load_model_mlflow.py
   
   → Affiche les modèles dans le registry
   → Teste le chargement
   
   python3 load_model_mlflow.py info
   → Affiche infos détaillées de toutes les versions

5️⃣  PIPELINE DVC (optionnel)
   dvc repro
   
   → Exécute le pipeline complet
   → Reproduit les résultats

📁 FICHIERS CRÉÉS:

MLOps:
  ✓ train_with_mlflow.py     → Script d'entraînement
  ✓ main_mlflow.py            → Streamlit avec MLflow
  ✓ load_model_mlflow.py      → Utilitaire modèles
  ✓ params.yaml               → Hyperparamètres
  ✓ dvc.yaml                  → Pipeline DVC
  ✓ requirements.txt          → Dépendances

Artifacts (générés après training):
  ✓ car_model.pkl
  ✓ scaler.pkl
  ✓ feature_info.json
  ✓ price_scaler_info.json
  ✓ encoders.pkl
  ✓ feature_importance.csv/png
  ✓ predictions_plot.png
  ✓ residuals_plot.png

📊 CE QUE MLFLOW TRACK:

Métriques:
  • R² Score (train/test)
  • MSE, RMSE, MAE
  • Feature Importance

Hyperparamètres:
  • n_estimators, max_depth, etc.
  • test_size, random_state

Artifacts:
  • Modèles (.pkl)
  • Plots (PNG)
  • Feature importance (CSV)

🎓 POUR VOTRE PROFESSEUR:

Ce projet démontre:
  ✅ MLflow: Tracking expériences + Model Registry
  ✅ DVC: Version control données/modèles
  ✅ Pipeline reproductible
  ✅ Interface utilisateur (Streamlit)
  ✅ Documentation complète

📖 DOCUMENTATION COMPLÈTE:
   Voir README_MLOPS.md

🚀 COMMENCEZ PAR:
   1. python3 train_with_mlflow.py
   2. mlflow ui  (dans un nouveau terminal)
   3. streamlit run main_mlflow.py  (dans un autre terminal)

💡 BESOIN D'AIDE?
   - Consultez README_MLOPS.md
   - Tous les scripts ont des docstrings
   - MLflow UI est très intuitif
""")
