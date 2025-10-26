"""
Module endpoints
----------------
Définit les routes de l'API pour l'inférence, l'upload de données, etc.
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def healthcheck():
    """Endpoint de santé."""
    return {"health": "green"}
