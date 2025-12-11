"""
Module main (FastAPI)
---------------------
Point d'entrée de l'API. Lance le serveur et définit la configuration générale.
"""
import uvicorn
from endpoints import router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Deep Learning API", version="0.1.0")
app.include_router(router, prefix="/api")


# Autorise le frontend local (Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    """Endpoint racine pour vérifier l'état."""
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",           # Chemin vers l'app (module:variable)
        host="0.0.0.0",               # Adresse d'écoute
        port=8000,                    # Port d'écoute
        reload=True,                  # Redémarrage auto à chaque changement de code
        log_level="info",             # Niveau de logs (debug, info, warning, error, critical)
        workers=1,                    # Nombre de workers (processus)
        access_log=True,              # Afficher les logs d'accès HTTP
        use_colors=True,              # Couleurs dans les logs
        proxy_headers=True,           # Support des headers proxy (X-Forwarded-For, etc.)
        forwarded_allow_ips="*",      # IPs autorisées pour les headers proxy
        timeout_keep_alive=5,         # Timeout keep-alive (secondes)
        limit_concurrency=None,       # Limite de connexions simultanées
        limit_max_requests=None,      # Limite de requêtes par worker
        ssl_keyfile=None,             # Fichier clé SSL (pour HTTPS)
        ssl_certfile=None,            # Fichier certificat SSL (pour HTTPS)
        ssl_ca_certs=None,            # Fichier CA (pour HTTPS)
        root_path="",                 # Préfixe d'URL si l'app n'est pas à la racine
        server_header=True,           # En-tête Server dans la réponse
        date_header=True,             # En-tête Date dans la réponse
        lifespan="auto",              # Gestion du cycle de vie (auto, on, off)
    )
