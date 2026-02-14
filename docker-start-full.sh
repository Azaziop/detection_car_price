#!/bin/bash

echo "🚀 Démarrage de la stack MLOps complète..."
echo "================================================"

# Arrêter l'ancien Airflow Docker si running
echo "🛑 Arrêt de l'ancien Airflow Docker..."
cd airflow-docker 2>/dev/null && docker compose down 2>/dev/null
cd ..

# Arrêter les services locaux
echo "🛑 Arrêt des services locaux..."
pkill -f "streamlit run" 2>/dev/null || true
pkill -f "mlflow server" 2>/dev/null || true
pkill -f "airflow" 2>/dev/null || true

# Configurer Docker PATH
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"

# Build et démarrage
echo "🐳 Construction des images Docker..."
docker compose -f docker-compose-full.yml build

echo "🚀 Démarrage des services..."
docker compose -f docker-compose-full.yml up -d

echo ""
echo "⏳ Attente de l'initialisation (30 secondes)..."
sleep 30

echo ""
echo "👤 Vérification des utilisateurs Airflow..."
docker compose -f docker-compose-full.yml exec airflow-webserver airflow users list

echo ""
echo "================================================"
echo "✅ Stack MLOps démarrée avec succès!"
echo "================================================"
echo ""
echo "📊 Services disponibles:"
echo "  • MLflow:    http://localhost:5000"
echo "  • Streamlit: http://localhost:8501"
echo "  • Airflow:   http://localhost:8080"
echo ""
echo "📝 Identifiants Airflow:"
echo "  Username: admin"
echo "  Password: admin"
echo ""
echo "🔧 Commandes utiles:"
echo "  • Logs:      docker compose -f docker-compose-full.yml logs -f"
echo "  • Status:    docker compose -f docker-compose-full.yml ps"
echo "  • Arrêter:   docker compose -f docker-compose-full.yml down"
echo ""
docker compose -f docker-compose-full.yml ps
