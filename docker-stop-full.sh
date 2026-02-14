#!/bin/bash

echo "🛑 Arrêt de la stack MLOps complète..."
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
docker compose -f docker-compose-full.yml down

echo "✅ Tous les services sont arrêtés"
