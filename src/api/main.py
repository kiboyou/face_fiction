"""
Module main (FastAPI)
---------------------
Point d'entrée de l'API. Lance le serveur et définit la configuration générale.
"""
from fastapi import FastAPI

from .endpoints import router

app = FastAPI(title="Deep Learning API", version="0.1.0")
app.include_router(router, prefix="/api")


@app.get("/")
def root():
    """Endpoint racine pour vérifier l'état."""
    return {"status": "ok"}
