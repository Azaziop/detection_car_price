# Gunicorn configuration for MLflow
# Disable host header checking to allow Docker internal networking

bind = "0.0.0.0:5000"
workers = 4
timeout = 300
forwarded_allow_ips = '*'

# Trust all proxy headers
proxy_protocol = False
proxy_allow_ips = '*'
