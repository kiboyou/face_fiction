# FaceFiction — Détection de DeepFakes

Ce dépôt contient le projet FaceFiction : une application de bout en bout pour la détection de deepfakes (classification binaire Real / Fake). Le projet combine recherche ML, MLOps, backend API et une interface frontend pour la démonstration et l'évaluation.
Principaux objectifs
- Construire une pipeline de détection de deepfakes reproductible.
- Suivre les expériences et modèles avec MLflow.
- Exposer un modèle via une API FastAPI et fournir un frontend React + TypeScript pour la démonstration.
- Évaluer la robustesse cross-dataset et l'interprétabilité (Grad-CAM, SHAP).
Stack technique (option avancée)
- Frontend : React.js + TypeScript
- Backend : FastAPI
- Deep Learning : PyTorch / torchvision (ou équivalent)
- MLOps : MLflow
- Containerisation : Docker & docker-compose
Dataset recommandé
- DeepFake Detection Challenge (DFDC) — préparer scripts de téléchargement et préprocessing.
Métriques clés
- AUC (ROC)
- Accuracy
- Robustesse cross-dataset (évaluer sur jeux externes)
Planification (6 semaines)

Semaine 1 — EDA & Baseline
- Exploration des données et statistiques.
- Baseline rapide pour valider pipeline d'entraînement.
- Configuration initiale de MLflow.
Semaine 2 — Modélisation & Tracking
- Implémentation de modèles avancés : EfficientNet, ResNet, ViT.
- Preprocessing, augmentation, et tracking systématique avec MLflow.
Semaine 3 — Expérimentation & Robustesse
- Optimisation d'hyperparamètres (Optuna recommandé).
- Évaluations cross-dataset et interprétabilité (Grad-CAM, SHAP).
Semaine 4 — API Backend & Docker
- Déploiement d'une API FastAPI pour servir le modèle.
- Tests unitaires, contractuels et containerisation (Docker).
Semaine 5 — Frontend & Intégration
- Frontend React + TypeScript pour visualiser prédictions et explications.
- Intégration continue backend-frontend.
Semaine 6 — Tests, Documentation & Éthique
- Tests E2E, documentation, préparation de la démonstration.
- Rapport sur questions éthiques et limites du modèle.
Critères de réussite
- Robustesse : évaluations cross-dataset (pondération importante).
- Stack Full-Stack : intégration complète React/TS + FastAPI.
- Interprétabilité : Grad-CAM/SHAP appliqués aux prédictions.
- Analyse éthique et gouvernance des risques.
Organisation du dépôt (vue rapide)
- `frontend/` : application React + TypeScript (interface de démo).
- `src/` : code backend (FastAPI), scripts ML, preprocessing, modèles.
- `notebooks/` : notebooks pour EDA, entraînement et évaluation.
- `mlruns/` : répertoire MLflow local (suivi des expériences).
- `tests/` : tests unitaires et d'intégration.
- `docker/` : Dockerfiles et `docker-compose.yml` pour orchestration.
Comment démarrer (dev rapide)

1) Installer les dépendances Python (virtuel conseillé)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Lancer le backend (FastAPI) en local

```bash
# depuis la racine
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

3) Lancer le frontend en local

```bash
cd frontend
npm install
npm run dev
```

4) MLflow (tracking)

```bash
# Démarrer le serveur MLflow local
mlflow ui --backend-store-uri ./mlruns --port 5000
```

5) Docker (optionnel : démarrage tout-en-un)

```bash
docker compose up --build
```

Contribuer
- Les contributions sont les bienvenues. Avant de soumettre une PR, documentez vos changements et ajoutez/mettre à jour les tests.

Aspects éthiques
- Le projet traite des deepfakes — réfléchissez aux usages, risques de mésinformation, vie privée et respect du consentement. Inclure un document `ETHICS.md` (à ajouter) pour détailler les analyses et mesures d'atténuation.

Contact
- Voir le dossier `frontend/src/app/api/contact` ou l'endpoint FastAPI `POST /contact` pour informations de contact.

Licence
- (À préciser) — ajoutez une licence appropriée (MIT, Apache-2.0, etc.) si le projet est destiné à être partagé.

---
Cette version du README a été adaptée à partir du cahier des charges fourni pour FaceFiction. Elle doit évoluer au fur et à mesure que le projet avance (scripts d'installation, commandes Docker, variables d'environnement détaillées, etc.).
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