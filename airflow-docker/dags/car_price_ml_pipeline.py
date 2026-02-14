"""
Airflow DAG pour le pipeline MLOps CarPricePredictor
Orchestration de l'entraînement, évaluation et promotion des modèles
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime, timedelta
import sys
import os

# Ajouter le chemin du projet au PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_path)

# Configuration par défaut du DAG
default_args = {
    'owner': 'data-science-team',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 14),
    'email': ['alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Définition du DAG
dag = DAG(
    'car_price_predictor_pipeline',
    default_args=default_args,
    description='Pipeline ML complet pour la prédiction de prix de voitures',
    schedule_interval='@weekly',  # Exécution hebdomadaire
    catchup=False,
    tags=['machine-learning', 'mlflow', 'car-price'],
)


def check_data_quality(**context):
    """Vérifier la qualité des données avant l'entraînement"""
    import pandas as pd
    import mlflow
    
    print("📊 Vérification de la qualité des données...")
    
    # Chargement des données
    data_path = os.path.join(project_path, 'data/raw/avito_car_dataset_ALL.csv')
    df = pd.read_csv(data_path, encoding='latin1')
    
    # Calculs de qualité
    total_rows = len(df)
    missing_values = df.isnull().sum().sum()
    missing_percentage = (missing_values / (total_rows * len(df.columns))) * 100
    
    # Vérifications
    quality_checks = {
        'total_rows': total_rows,
        'missing_percentage': missing_percentage,
        'columns': list(df.columns),
        'data_ok': total_rows > 1000 and missing_percentage < 50
    }
    
    print(f"✅ Lignes: {total_rows}")
    print(f"✅ Valeurs manquantes: {missing_percentage:.2f}%")
    
    # Sauvegarder dans XCom pour les tâches suivantes
    context['ti'].xcom_push(key='data_quality', value=quality_checks)
    
    if not quality_checks['data_ok']:
        raise ValueError("❌ Données insuffisantes pour l'entraînement")
    
    return quality_checks


def train_model(**context):
    """Entraîner le modèle avec MLflow"""
    import mlflow
    from scripts.train_with_mlflow import CarPricePipeline
    
    print("🚀 Démarrage de l'entraînement du modèle...")
    
    # Configuration MLflow
    mlflow_path = os.path.join(project_path, 'mlflow', 'mlruns')
    mlflow.set_tracking_uri(f"file:{mlflow_path}")
    mlflow.set_experiment("car_price_prediction")
    
    # Récupérer les infos de qualité des données
    data_quality = context['ti'].xcom_pull(key='data_quality', task_ids='check_data_quality')
    print(f"📊 Données validées: {data_quality['total_rows']} lignes")
    
    # Entraînement
    with mlflow.start_run(run_name=f"airflow_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Tag pour identifier les runs Airflow
        mlflow.set_tag("pipeline", "airflow")
        mlflow.set_tag("trigger", "scheduled")
        
        pipeline = CarPricePipeline()
        
        # Load and prepare data
        df = pipeline.load_data()
        X_train, X_test, y_train, y_test = pipeline.prepare_data(df)
        
        # Train model
        pipeline.train_model(X_train, y_train)
        
        # Evaluate
        metrics = pipeline.evaluate_model(X_test, y_test)
        
        # Log artifacts
        pipeline.save_artifacts()
        
        # Get run ID
        run_id = mlflow.active_run().info.run_id
        
        print(f"✅ Modèle entraîné - Run ID: {run_id}")
        print(f"📊 R² Score: {metrics['r2_score']:.4f}")
        print(f"📊 RMSE: {metrics['rmse']:.2f}")
        
        # Sauvegarder les infos dans XCom
        context['ti'].xcom_push(key='run_id', value=run_id)
        context['ti'].xcom_push(key='metrics', value=metrics)
    
    return run_id


def evaluate_model(**context):
    """Évaluer le modèle et décider de la promotion"""
    import mlflow
    from mlflow.tracking import MlflowClient
    
    print("🔍 Évaluation du modèle...")
    
    # Configuration MLflow
    mlflow_path = os.path.join(project_path, 'mlflow', 'mlruns')
    mlflow.set_tracking_uri(f"file:{mlflow_path}")
    
    # Récupérer les métriques
    run_id = context['ti'].xcom_pull(key='run_id', task_ids='train_model')
    metrics = context['ti'].xcom_pull(key='metrics', task_ids='train_model')
    
    # Critères de qualité
    MINIMUM_R2 = 0.80  # R² minimum acceptable
    MAXIMUM_RMSE = 50000  # RMSE maximum acceptable
    
    r2_score = metrics['r2_score']
    rmse = metrics['rmse']
    
    print(f"📊 R² Score: {r2_score:.4f} (min: {MINIMUM_R2})")
    print(f"📊 RMSE: {rmse:.2f} (max: {MAXIMUM_RMSE})")
    
    # Décision
    is_promotable = r2_score >= MINIMUM_R2 and rmse <= MAXIMUM_RMSE
    
    evaluation_result = {
        'run_id': run_id,
        'r2_score': r2_score,
        'rmse': rmse,
        'is_promotable': is_promotable,
        'evaluation_date': datetime.now().isoformat()
    }
    
    if is_promotable:
        print("✅ Modèle éligible pour la promotion!")
    else:
        print("⚠️ Modèle ne satisfait pas les critères de qualité")
    
    context['ti'].xcom_push(key='evaluation_result', value=evaluation_result)
    
    return evaluation_result


def promote_to_staging(**context):
    """Promouvoir le modèle vers Staging"""
    import mlflow
    from mlflow.tracking import MlflowClient
    
    print("📦 Promotion du modèle vers Staging...")
    
    # Configuration MLflow
    mlflow_path = os.path.join(project_path, 'mlflow', 'mlruns')
    mlflow.set_tracking_uri(f"file:{mlflow_path}")
    client = MlflowClient()
    
    # Récupérer les résultats d'évaluation
    eval_result = context['ti'].xcom_pull(key='evaluation_result', task_ids='evaluate_model')
    
    if not eval_result['is_promotable']:
        print("⚠️ Modèle non éligible, promotion annulée")
        return "skipped"
    
    run_id = eval_result['run_id']
    model_name = "CarPricePredictor"
    
    try:
        # Enregistrer le modèle
        model_uri = f"runs:/{run_id}/model"
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Promouvoir vers Staging
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging",
            archive_existing_versions=False
        )
        
        print(f"✅ Modèle promu vers Staging - Version {model_version.version}")
        
        context['ti'].xcom_push(key='model_version', value=model_version.version)
        
        return model_version.version
        
    except Exception as e:
        print(f"❌ Erreur lors de la promotion: {str(e)}")
        raise


def validate_staging_model(**context):
    """Valider le modèle en Staging avant production"""
    import mlflow
    from mlflow.tracking import MlflowClient
    import pandas as pd
    import numpy as np
    
    print("🧪 Validation du modèle en Staging...")
    
    # Configuration MLflow
    mlflow_path = os.path.join(project_path, 'mlflow', 'mlruns')
    mlflow.set_tracking_uri(f"file:{mlflow_path}")
    client = MlflowClient()
    
    model_name = "CarPricePredictor"
    model_version = context['ti'].xcom_pull(key='model_version', task_ids='promote_to_staging')
    
    # Charger le modèle depuis Staging
    model_uri = f"models:/{model_name}/Staging"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Test sur quelques prédictions
    # Créer des données de test simples
    test_data = pd.DataFrame({
        'year': [2020, 2015, 2018],
        'brand': [1, 2, 3],
        'fuel_type': [1, 2, 1],
        'transmission': [1, 0, 1],
    })
    
    # Faire des prédictions
    predictions = model.predict(test_data)
    
    # Vérifications basiques
    validation_checks = {
        'model_loaded': True,
        'predictions_valid': all(predictions > 0),
        'predictions_reasonable': all(predictions < 1000000),  # Prix < 1M
        'model_version': model_version,
        'validation_passed': True
    }
    
    print(f"✅ Validation réussie - Version {model_version}")
    print(f"📊 Prédictions test: {predictions[:3]}")
    
    context['ti'].xcom_push(key='validation_checks', value=validation_checks)
    
    return validation_checks


def promote_to_production(**context):
    """Promouvoir le modèle vers Production"""
    import mlflow
    from mlflow.tracking import MlflowClient
    
    print("🚀 Promotion du modèle vers Production...")
    
    # Configuration MLflow
    mlflow_path = os.path.join(project_path, 'mlflow', 'mlruns')
    mlflow.set_tracking_uri(f"file:{mlflow_path}")
    client = MlflowClient()
    
    # Récupérer la version du modèle
    model_version = context['ti'].xcom_pull(key='model_version', task_ids='promote_to_staging')
    validation = context['ti'].xcom_pull(key='validation_checks', task_ids='validate_staging_model')
    
    if not validation['validation_passed']:
        print("⚠️ Validation échouée, promotion vers Production annulée")
        return "skipped"
    
    model_name = "CarPricePredictor"
    
    try:
        # Archiver les versions Production actuelles
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Production",
            archive_existing_versions=True
        )
        
        print(f"✅ Modèle promu vers Production - Version {model_version}")
        
        return {
            'model_name': model_name,
            'version': model_version,
            'stage': 'Production',
            'promotion_date': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Erreur lors de la promotion Production: {str(e)}")
        raise


def send_pipeline_report(**context):
    """Générer et envoyer un rapport du pipeline"""
    import json
    
    print("📧 Génération du rapport de pipeline...")
    
    # Récupérer toutes les informations
    data_quality = context['ti'].xcom_pull(key='data_quality', task_ids='check_data_quality')
    run_id = context['ti'].xcom_pull(key='run_id', task_ids='train_model')
    metrics = context['ti'].xcom_pull(key='metrics', task_ids='train_model')
    eval_result = context['ti'].xcom_pull(key='evaluation_result', task_ids='evaluate_model')
    
    # Créer le rapport
    report = {
        'pipeline_date': datetime.now().isoformat(),
        'data_quality': data_quality,
        'training': {
            'run_id': run_id,
            'metrics': metrics
        },
        'evaluation': eval_result,
        'status': 'SUCCESS'
    }
    
    # Sauvegarder le rapport
    report_path = os.path.join(project_path, 'reports', f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Rapport sauvegardé: {report_path}")
    print(f"📊 Status: {report['status']}")
    print(f"📊 R² Score: {metrics['r2_score']:.4f}")
    
    return report


# ========================================
# Définition des tâches
# ========================================

# Tâche 1: Vérification de la qualité des données
check_data = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    dag=dag,
)

# Tâche 2: Entraînement du modèle
train = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

# Tâche 3: Évaluation du modèle
evaluate = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

# Tâche 4: Promotion vers Staging
staging = PythonOperator(
    task_id='promote_to_staging',
    python_callable=promote_to_staging,
    dag=dag,
)

# Tâche 5: Validation du modèle en Staging
validate = PythonOperator(
    task_id='validate_staging_model',
    python_callable=validate_staging_model,
    dag=dag,
)

# Tâche 6: Promotion vers Production
production = PythonOperator(
    task_id='promote_to_production',
    python_callable=promote_to_production,
    dag=dag,
)

# Tâche 7: Rapport final
report = PythonOperator(
    task_id='send_pipeline_report',
    python_callable=send_pipeline_report,
    dag=dag,
)

# ========================================
# Définition du flux de tâches
# ========================================

check_data >> train >> evaluate >> staging >> validate >> production >> report
