# Projet Deep Learning - Terminale Data Science

Ce dépôt propose une architecture technique recommandée pour un projet de Deep Learning.

## Structure principale

- src/
  - data/
    - __init__.py
    - dataset.py
    - preprocessing.py
  - models/
    - __init__.py
    - architecture.py
    - training.py
  - api/
    - __init__.py
    - main.py
    - endpoints.py
  - utils/
    - __init__.py
    - config.py
    - metrics.py
- frontend/ (optionnel)
  - src/
  - public/
  - package.json
  - README.md
- streamlit_app/ (optionnel)
  - app.py
  - pages/
  - components/
- notebooks/
  - 01_data_exploration.ipynb
  - 02_model_training.ipynb
  - 03_evaluation.ipynb
- tests/
  - test_data.py
  - test_models.py
  - test_api.py
- docker/
  - Dockerfile.api
  - Dockerfile.frontend (optionnel)
  - docker-compose.yml
- mlruns/
- requirements.txt
- .gitignore

## Démarrage rapide

1. Créer et activer l'environnement Python:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Lancer l'API FastAPI en local (sans Docker):

```
uvicorn src.api.main:app --reload --port 8000
```

```

1. (Optionnel) Avec Docker:

```
docker compose -f docker/docker-compose.yml up --build
```