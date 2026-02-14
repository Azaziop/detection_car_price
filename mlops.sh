#!/bin/bash

# Script d'aide pour gérer la stack MLOps
# Usage: ./mlops.sh [command]

# Couleurs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configurer Docker PATH
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"

case "$1" in
    streamlit)
        echo -e "${BLUE}🎨 Démarrage de Streamlit...${NC}"
        source .venv/bin/activate
        streamlit run main_mlflow.py
        ;;
    
    mlflow)
        echo -e "${BLUE}📊 Démarrage de MLflow UI...${NC}"
        source .venv/bin/activate
        mlflow ui
        ;;
    
    train)
        echo -e "${BLUE}🚀 Entraînement du modèle...${NC}"
        source .venv/bin/activate
        python scripts/train_with_mlflow.py
        ;;
    
    airflow-start)
        echo -e "${BLUE}🔄 Démarrage d'Airflow (Docker)...${NC}"
        cd airflow-docker
        docker compose up -d
        echo -e "${GREEN}✅ Airflow démarré sur http://localhost:8080${NC}"
        echo -e "${YELLOW}   Login: airflow / airflow${NC}"
        ;;
    
    airflow-stop)
        echo -e "${BLUE}🛑 Arrêt d'Airflow...${NC}"
        cd airflow-docker
        docker compose down
        echo -e "${GREEN}✅ Airflow arrêté${NC}"
        ;;
    
    airflow-logs)
        echo -e "${BLUE}📋 Logs Airflow...${NC}"
        cd airflow-docker
        docker compose logs -f
        ;;
    
    airflow-restart)
        echo -e "${BLUE}🔄 Redémarrage d'Airflow...${NC}"
        cd airflow-docker
        docker compose restart
        echo -e "${GREEN}✅ Airflow redémarré${NC}"
        ;;
    
    test)
        echo -e "${BLUE}🧪 Exécution des tests...${NC}"
        source .venv/bin/activate
        pytest tests/ -v
        ;;
    
    test-cov)
        echo -e "${BLUE}🧪 Tests avec coverage...${NC}"
        source .venv/bin/activate
        pytest tests/ --cov=. --cov-report=html
        echo -e "${GREEN}✅ Rapport dans reports/htmlcov/index.html${NC}"
        ;;
    
    status)
        echo -e "${BLUE}📊 Statut de la stack MLOps${NC}"
        echo ""
        
        # Python
        echo -e "${GREEN}Python:${NC}"
        source .venv/bin/activate
        python --version
        echo ""
        
        # MLflow
        echo -e "${GREEN}MLflow:${NC}"
        python -c "import mlflow; print(f'  Version: {mlflow.__version__}')"
        echo ""
        
        # Streamlit
        echo -e "${GREEN}Streamlit:${NC}"
        python -c "import streamlit; print(f'  Version: {streamlit.__version__}')"
        echo ""
        
        # Airflow
        echo -e "${GREEN}Airflow (Docker):${NC}"
        cd airflow-docker
        if docker compose ps | grep -q "Up"; then
            echo -e "  ${GREEN}✅ Running${NC}"
            docker compose ps
        else
            echo -e "  ${YELLOW}⏸️  Stopped${NC}"
        fi
        ;;
    
    open-airflow)
        echo -e "${BLUE}🌐 Ouverture d'Airflow...${NC}"
        open http://localhost:8080
        ;;
    
    open-mlflow)
        echo -e "${BLUE}🌐 Ouverture de MLflow...${NC}"
        open http://localhost:5000
        ;;
    
    open-streamlit)
        echo -e "${BLUE}🌐 Ouverture de Streamlit...${NC}"
        open http://localhost:8501
        ;;
    
    *)
        echo -e "${BLUE}🚗 MLOps Stack - Prédicteur Prix Voiture${NC}"
        echo ""
        echo "Usage: ./mlops.sh [command]"
        echo ""
        echo "Commandes disponibles:"
        echo ""
        echo -e "${GREEN}Interface & Applications:${NC}"
        echo "  streamlit          - Démarrer l'application Streamlit"
        echo "  mlflow             - Démarrer l'UI MLflow"
        echo "  train              - Entraîner le modèle"
        echo ""
        echo -e "${GREEN}Airflow (Docker):${NC}"
        echo "  airflow-start      - Démarrer Airflow"
        echo "  airflow-stop       - Arrêter Airflow"
        echo "  airflow-restart    - Redémarrer Airflow"
        echo "  airflow-logs       - Voir les logs"
        echo ""
        echo -e "${GREEN}Tests:${NC}"
        echo "  test               - Lancer les tests unitaires"
        echo "  test-cov           - Tests avec coverage"
        echo ""
        echo -e "${GREEN}Monitoring:${NC}"
        echo "  status             - Afficher le statut de la stack"
        echo "  open-airflow       - Ouvrir Airflow dans le navigateur"
        echo "  open-mlflow        - Ouvrir MLflow dans le navigateur"
        echo "  open-streamlit     - Ouvrir Streamlit dans le navigateur"
        echo ""
        echo -e "${YELLOW}Exemples:${NC}"
        echo "  ./mlops.sh streamlit        # Lance l'app"
        echo "  ./mlops.sh airflow-start    # Démarre Airflow"
        echo "  ./mlops.sh status           # Vérifie tout"
        ;;
esac
