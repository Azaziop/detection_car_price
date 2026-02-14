#!/bin/bash
# Script pour initialiser et démarrer Apache Airflow
# Usage: ./start_airflow.sh

echo "🚀 Initialisation d'Apache Airflow pour CarPricePredictor Pipeline"
echo "=================================================================="

# Définir les variables d'environnement
export AIRFLOW_HOME="$(pwd)/airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$(pwd)/airflow/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES="False"

# Couleurs pour les messages
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Installer les dépendances si nécessaire
echo -e "${YELLOW}📦 Vérification des dépendances...${NC}"
if ! python -c "import airflow" 2>/dev/null; then
    echo "Installation d'Apache Airflow..."
    pip install -r requirements.txt
else
    echo -e "${GREEN}✅ Apache Airflow déjà installé${NC}"
fi

# 2. Initialiser la base de données Airflow
echo -e "${YELLOW}🗄️  Initialisation de la base de données Airflow...${NC}"
if [ ! -f "$AIRFLOW_HOME/airflow.db" ]; then
    airflow db init
    echo -e "${GREEN}✅ Base de données initialisée${NC}"
else
    echo -e "${GREEN}✅ Base de données déjà existante${NC}"
fi

# 3. Créer un utilisateur admin si nécessaire
echo -e "${YELLOW}👤 Configuration de l'utilisateur admin...${NC}"
airflow users list | grep -q "admin" || airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

echo -e "${GREEN}✅ Utilisateur admin configuré (username: admin, password: admin)${NC}"

# 4. Vérifier que les DAGs sont détectés
echo -e "${YELLOW}📂 Vérification des DAGs...${NC}"
airflow dags list | grep -q "car_price_predictor_pipeline"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ DAG 'car_price_predictor_pipeline' détecté${NC}"
else
    echo -e "${YELLOW}⚠️  DAG pas encore détecté, attendez quelques secondes...${NC}"
fi

# 5. Instructions pour démarrer Airflow
echo ""
echo "=================================================================="
echo -e "${GREEN}🎉 Airflow est prêt!${NC}"
echo "=================================================================="
echo ""
echo -e "${YELLOW}🚀 Démarrage recommandé - Mode Standalone:${NC}"
echo "  export AIRFLOW_HOME=$(pwd)/airflow"
echo "  airflow standalone"
echo ""
echo -e "${YELLOW}📊 Alternative - Scheduler seul (sans interface web):${NC}"
echo "  export AIRFLOW_HOME=$(pwd)/airflow"
echo "  airflow scheduler"
echo ""
echo "=================================================================="
echo -e "${GREEN}📊 Interface Web:${NC} http://localhost:8080"
echo -e "${GREEN}👤 Login:${NC} Consulter le terminal standalone"
echo "=================================================================="
echo ""
echo -e "${YELLOW}💡 Commandes utiles:${NC}"
echo "  - Liste des DAGs:        airflow dags list"
echo "  - Tester un DAG:         airflow dags test car_price_predictor_pipeline $(date +%Y-%m-%d)"
echo "  - Activer un DAG:        airflow dags unpause car_price_predictor_pipeline"
echo "  - Logs d'une tâche:      airflow tasks logs car_price_predictor_pipeline train_model $(date +%Y-%m-%d)"
echo ""
