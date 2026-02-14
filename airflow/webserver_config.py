import os
from airflow import configuration as conf
from flask_appbuilder.security.manager import AUTH_DB

# Utiliser le worker sync au lieu de gunicorn pour éviter les SIGSEGV sur macOS ARM64
# Configuration du webserver
basedir = os.path.abspath(os.path.dirname(__file__))

# Configuration Flask
WTF_CSRF_ENABLED = True
SECRET_KEY = '\2\1thisismyscretkey\1\2\e\y\y\h'

# Configuration Auth
AUTH_TYPE = AUTH_DB
AUTH_ROLE_ADMIN = 'Admin'
AUTH_USER_REGISTRATION = False

# Configuration du serveur
# IMPORTANT: Utiliser un seul worker sync pour éviter les crashes SIGSEGV
GUNICORN_CMD_ARGS = '--workers=1 --worker-class=sync --timeout=120'
