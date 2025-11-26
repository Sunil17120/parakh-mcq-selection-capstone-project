# gunicorn_conf.py

import multiprocessing

# 1. Bind Address and Port
# This tells Gunicorn to listen on all interfaces (0.0.0.0) on port 8000.
bind = '0.0.0.0:8000'

# 2. Worker Class (Crucial for FastAPI)
# Use uvicorn worker class for handling asynchronous requests (ASGI server)
worker_class = 'uvicorn.workers.UvicornWorker'

# 3. Worker Count (Performance Optimization)
# A common heuristic is (2 * CPU_cores) + 1. 
# This helps maximize throughput while avoiding contention.
# We use multiprocessing to detect the host machine's core count automatically.
workers = 2

# 4. Application Entry Point
# The format is <module_name>:<fastapi_app_instance>
# In your case: main.py -> app
wsgi_app = 'main:app' 

# 5. Logging
# Direct logs to standard output/error so they appear in your Docker/Cloud logs
loglevel = 'info'
accesslog = '-' # stdout
errorlog = '-'  # stderr