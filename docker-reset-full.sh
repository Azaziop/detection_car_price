#!/bin/bash

echo "🔄 Réinitialisation complète de la stack..."
echo "=========================================="

export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"

# Arrêter tous les services
echo "1️⃣ Arrêt des services..."
docker compose -f docker-compose-full.yml down

# Supprimer les volumes
echo "2️⃣ Suppression des volumes..."
docker volume rm pythonproject9_postgres-db-volume 2>/dev/null || true

# Nettoyer les logs Airflow
echo "3️⃣ Nettoyage des logs..."
rm -rf airflow/logs/* airflow/airflow.cfg 2>/dev/null || true

# Redémarrer
echo "4️⃣ Redémarrage..."
docker compose -f docker-compose-full.yml up -d

echo ""
echo "⏳ Attente de l'initialisation (30 secondes)..."
sleep 30

echo ""
echo "5️⃣ Vérification des utilisateurs..."
docker compose -f docker-compose-full.yml exec airflow-webserver airflow users list

echo ""
echo "================================================"
echo "✅ Réinitialisation terminée!"
echo "================================================"
echo ""
echo "📝 Identifiants Airflow:"
echo "  Username: admin"
echo "  Password: admin"
echo ""
echo "🌐 Interface: http://localhost:8080"
echo ""
