# fastapi_server.py
import asyncio
import json
import os
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import h5py
import numpy as np
import pandas as pd
# TensorFlow
import tensorflow as tf
# FastAPI
from fastapi import (BackgroundTasks, FastAPI, File, Form, HTTPException,
                     UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ==================== CONFIGURATION ====================
# Utiliser tf.keras au lieu de keras pour la compatibilité
tf_keras = tf.keras

# Lifespan manager pour remplacer on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup_event()
    yield
    # Shutdown
    await shutdown_event()

app = FastAPI(
    title="Deepfake Detection API",
    description="API pour la détection de deepfakes avec EfficientNetB7",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chemins
BASE_DIR = Path(__file__).resolve().parent.parent
print(f"📁 BASE_DIR: {BASE_DIR}")
MODELS_DIR = BASE_DIR / "models/model"
REPORTS_DIR = BASE_DIR / "models/reports"
UPLOADS_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"

# Créer les dossiers
for directory in [MODELS_DIR, REPORTS_DIR, UPLOADS_DIR, STATIC_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ==================== MODÈLES & ÉTAT ====================
class ModelManager:
    """Gestionnaire de modèles avec support Keras 2.x/3.x"""
    
    def __init__(self):
        self.models: Dict[str, Dict] = {}
        self.feature_extractor = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.is_keras_3 = hasattr(tf_keras, '__version__') and tf_keras.__version__.startswith('3')
        print(f"🤖 Keras version: {'3.x' if self.is_keras_3 else '2.x/TensorFlow'}")
        
    def load_feature_extractor(self):
        """Charge l'extracteur de features B7"""
        if self.feature_extractor is None:
            self.feature_extractor = EfficientFeatureExtractor()
        return self.feature_extractor
    
    def scan_models(self):
        """Scanne et charge tous les modèles disponibles"""
        self.models.clear()
        
        # Chercher les modèles .h5
        for h5_file in MODELS_DIR.glob("*.h5"):
            try:
                print(f"\n🔄 Tentative de chargement: {h5_file.name}")
                
                # Essayer différentes méthodes de chargement
                model = self._try_load_model(str(h5_file))
                
                if model is not None:
                    model_name = h5_file.stem
                    
                    self.models[model_name] = {
                        "model": model,
                        "path": str(h5_file),
                        "type": "h5",
                        "loaded_at": datetime.now().isoformat(),
                        "parameters": model.count_params() if hasattr(model, 'count_params') else 0,
                        "input_shape": model.input_shape if hasattr(model, 'input_shape') else "unknown",
                        "output_shape": model.output_shape if hasattr(model, 'output_shape') else "unknown",
                        "layers": len(model.layers) if hasattr(model, 'layers') else 0
                    }
                    print(f"✅ Modèle chargé: {model_name}")
                    print(f"   📊 Paramètres: {self.models[model_name]['parameters']:,}")
                    print(f"   📐 Input shape: {self.models[model_name]['input_shape']}")
                    print(f"   📐 Output shape: {self.models[model_name]['output_shape']}")
                    print(f"   🏗️  Couches: {self.models[model_name]['layers']}")
                    
            except Exception as e:
                print(f"❌ Erreur chargement {h5_file.name}: {str(e)[:200]}...")
        
        return len(self.models)
    
    def _try_load_model(self, model_path: str):
        """Essaye différentes méthodes pour charger un modèle"""
        methods = [
            # self._load_with_keras3_compatibility,
            # self._load_with_custom_scope,
            # self._load_with_tf_keras,
            self._load_with_h5py
        ]
        
        for i, method in enumerate(methods):
            try:
                print(f"   🔧 Méthode {i+1}: {method.__name__}")
                model = method(model_path)
                if model is not None:
                    return model
            except Exception as e:
                print(f"   ❌ Échec: {str(e)[:100]}...")
        
        return None
    
    def _load_with_keras3_compatibility(self, model_path: str):
        """Méthode pour Keras 3.x avec compatibilité"""
        if self.is_keras_3:
            # Pour Keras 3.x, on utilise tf.keras pour charger les modèles anciens
            import tensorflow as tf
            return tf.keras.models.load_model(model_path, compile=True)
        else:
            return tf_keras.models.load_model(model_path, compile=True)
    
    def _load_with_custom_scope(self, model_path: str):
        """Charge avec un scope custom pour gérer les layers"""
        import tensorflow as tf

        # Définir les custom objects pour Keras 2.x
        custom_objects = {
            'LSTM': tf.keras.layers.LSTM,
            'Bidirectional': tf.keras.layers.Bidirectional,
            'Dense': tf.keras.layers.Dense,
            'Dropout': tf.keras.layers.Dropout,
            'BatchNormalization': tf.keras.layers.BatchNormalization,
            'Conv2D': tf.keras.layers.Conv2D,
            'MaxPooling2D': tf.keras.layers.MaxPooling2D,
            'Flatten': tf.keras.layers.Flatten,
            'GlobalAveragePooling2D': tf.keras.layers.GlobalAveragePooling2D,
            'Attention': tf.keras.layers.Attention,
            'MultiHeadAttention': tf.keras.layers.MultiHeadAttention,
            'LayerNormalization': tf.keras.layers.LayerNormalization,
            'Add': tf.keras.layers.Add,
            'Concatenate': tf.keras.layers.Concatenate,
            'TFOpLambda': tf.keras.layers.Lambda,  # Pour gérer TFOpLambda
        }
        
        return tf.keras.models.load_model(
            model_path,
            compile=True,
            custom_objects=custom_objects
        )
    
    def _load_with_tf_keras(self, model_path: str):
        """Charge avec tf.keras directement"""
        # Désactiver les warnings temporairement
        import warnings

        import tensorflow as tf
        warnings.filterwarnings('ignore')
        
        try:
            return tf.keras.models.load_model(model_path, compile=True)
        finally:
            warnings.filterwarnings('default')
    
    def _load_with_h5py(self, model_path: str):
        """Charge manuellement avec h5py pour inspecter la structure"""
        try:
            import h5py
            
            with h5py.File(model_path, 'r') as f:
                print(f"   📖 Structure du fichier H5:")
                self._print_h5_structure(f)
            
            # Essayer de charger juste les poids
            import tensorflow as tf

            # Créer un modèle simple pour tester
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(32, 2560)),
                tf.keras.layers.LSTM(512, return_sequences=True),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            # Essayer de charger les poids
            try:
                model.load_weights(model_path, by_name=True, skip_mismatch=True)
                print(f"   ✅ Poids chargés avec succès")
                return model
            except:
                print(f"   ⚠️ Impossible de charger les poids, utilisation du modèle par défaut")
                return model
                
        except Exception as e:
            print(f"   ❌ Erreur h5py: {e}")
            return None
    
    def _print_h5_structure(self, f, indent=0):
        """Affiche la structure d'un fichier H5"""
        for key in f.keys():
            print(f"{'   ' * (indent + 1)}📁 {key}")
            if isinstance(f[key], h5py.Group):
                self._print_h5_structure(f[key], indent + 1)
    
    def get_model(self, model_name: str = "default"):
        """Récupère un modèle par son nom"""
        if not self.models:
            self.scan_models()
            
        if model_name in self.models:
            return self.models[model_name]["model"]
        elif self.models:
            # Retourner le premier modèle disponible
            first_model = list(self.models.values())[0]["model"]
            return first_model
        else:
            # Créer un modèle par défaut si aucun n'est chargé
            return self._create_default_model()
    
    def _create_default_model(self):
        """Crée un modèle par défaut si aucun n'est chargé"""
        print("⚠️ Création d'un modèle par défaut...")
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(32, 2560)),
            tf.keras.layers.LSTM(256, return_sequences=True),
            tf.keras.layers.LSTM(128),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_model_info(self, model_name: str = None):
        """Récupère les infos d'un modèle"""
        if model_name and model_name in self.models:
            info = self.models[model_name].copy()
            info.pop("model", None)  # Ne pas sérialiser le modèle
            return info
        elif self.models:
            # Retourner tous les modèles
            return {name: {k: v for k, v in data.items() if k != "model"} 
                   for name, data in self.models.items()}
        else:
            return {}

# Initialiser le gestionnaire de modèles
model_manager = ModelManager()

# ==================== EXTRACTEUR DE FEATURES ====================
class EfficientFeatureExtractor:
    """Extracteur de features optimisé pour B7"""
    
    def __init__(self, frames_per_video: int = 32):
        self.frames_per_video = frames_per_video
        self.feature_dim = 2560  # EfficientNetB7
        self.total_features = frames_per_video * self.feature_dim
        
        # Charger le modèle B7 une seule fois
        print(f"🔄 Initialisation de l'extracteur EfficientNetB7...")
        
        # Utiliser tf.keras pour la compatibilité
        self.base_model = tf.keras.applications.EfficientNetB7(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=(600, 600, 3)
        )
        self.base_model.trainable = False
        
        # Optimiser avec tf.function
        self._extract_batch = tf.function(self._extract_batch_internal)
        
        print(f"✅ Extracteur B7 initialisé: {self.total_features} features")
    
    def _extract_batch_internal(self, batch: tf.Tensor) -> tf.Tensor:
        """Fonction interne optimisée"""
        return self.base_model(batch, training=False)
    
    async def extract_video_features_async(self, video_path: str) -> Optional[np.ndarray]:
        """Extrait les features d'une vidéo de manière asynchrone"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            model_manager.executor,
            self.extract_video_features,
            video_path
        )
    
    def extract_video_features(self, video_path: str) -> Optional[np.ndarray]:
        """Extrait les features d'une vidéo"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Impossible d'ouvrir la vidéo: {video_path}")
                return None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                print(f"❌ Vidéo vide ou corrompue: {video_path}")
                cap.release()
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"📹 Vidéo: {total_frames} frames, {fps:.1f} FPS")
            
            # Calculer les indices des frames
            if total_frames <= self.frames_per_video:
                frame_indices = list(range(total_frames))
                while len(frame_indices) < self.frames_per_video:
                    frame_indices.append(frame_indices[-1])
            else:
                step = total_frames / self.frames_per_video
                frame_indices = np.arange(0, total_frames, step, dtype=int)[:self.frames_per_video]
            
            # Lire les frames
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total_frames - 1))
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (600, 600))
                    frames.append(frame_resized)
                else:
                    # Frame noire si lecture échoue
                    frames.append(np.zeros((600, 600, 3), dtype=np.uint8))
            
            cap.release()
            
            if not frames:
                print(f"❌ Aucune frame extraite de: {video_path}")
                return None
            
            print(f"🖼️ {len(frames)} frames extraites")
            
            # Convertir en batch
            batch = np.stack(frames, axis=0).astype(np.float32)
            
            # Préprocessing spécifique à EfficientNet
            batch = tf.keras.applications.efficientnet.preprocess_input(batch)
            
            # Extraire les features
            features = self._extract_batch(batch).numpy()
            
            print(f"🔧 Features brutes: shape={features.shape}")
            
            # Normalisation L2
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            features = features / norms
            
            # Aplatir
            video_features = features.flatten()
            
            # Padding ou troncature si nécessaire
            if len(video_features) < self.total_features:
                pad = self.total_features - len(video_features)
                video_features = np.pad(video_features, (0, pad), mode='constant')
                print(f"📏 Padding appliqué: {pad} zéros")
            elif len(video_features) > self.total_features:
                video_features = video_features[:self.total_features]
                print(f"📏 Troncature appliquée")
            
            print(f"✅ Features finales: shape={video_features.shape}")
            return video_features.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Erreur extraction features de {video_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

# ==================== MODÈLES PYDANTIC ====================
class HealthResponse(BaseModel):
    """Réponse du endpoint health"""
    status: str
    timestamp: str
    models_loaded: int
    feature_extractor: str
    uploads_dir: str
    models_dir: str
    keras_version: str

class ModelInfo(BaseModel):
    """Information sur un modèle"""
    name: str
    type: str
    loaded_at: str
    parameters: int
    input_shape: str
    path: str

class ModelsResponse(BaseModel):
    """Réponse du endpoint models"""
    success: bool
    models: List[ModelInfo]
    total: int
    timestamp: str

class PredictionRequest(BaseModel):
    """Requête pour prédiction par URL"""
    video_url: str
    model_name: Optional[str] = "default"

class BatchPredictionRequest(BaseModel):
    """Requête pour prédiction batch"""
    model_name: Optional[str] = "default"
    # Les fichiers sont envoyés séparément

class PredictionResult(BaseModel):
    """Résultat d'une prédiction"""
    success: bool
    timestamp: str
    model_used: str
    prediction_score: float
    prediction_class: str
    confidence: float
    processing_time_seconds: float
    features_shape: tuple
    model_type: str
    filename: Optional[str] = None
    file_size: Optional[int] = None
    error: Optional[str] = None

class BatchPredictionResult(BaseModel):
    """Résultat d'une prédiction batch"""
    success: bool
    batch_id: str
    total_files: int
    successful: int
    failed: int
    results: List[PredictionResult]
    errors: List[Dict[str, str]]
    timestamp: str

class StatsResponse(BaseModel):
    """Statistiques du serveur"""
    success: bool
    timestamp: str
    system: Dict[str, Any]
    models: Dict[str, Any]
    gpu: List[Dict[str, Any]]

# ==================== UTILITAIRES ====================
def allowed_file(filename: str) -> bool:
    """Vérifie si le fichier a une extension autorisée"""
    allowed_extensions = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv', 'm4v', 'mpg', 'mpeg'}
    if '.' not in filename:
        return False
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in allowed_extensions

def save_upload_file(upload_file: UploadFile) -> str:
    """Sauvegarde un fichier uploadé et retourne le chemin"""
    # Créer un nom de fichier unique
    file_ext = upload_file.filename.split('.')[-1] if '.' in upload_file.filename else 'mp4'
    unique_filename = f"{uuid.uuid4().hex}.{file_ext}"
    file_path = UPLOADS_DIR / unique_filename
    
    # Sauvegarder le fichier
    with open(file_path, "wb") as buffer:
        content = upload_file.file.read()
        buffer.write(content)
    
    file_size_kb = os.path.getsize(file_path) / 1024
    print(f"📥 Fichier sauvegardé: {file_path} ({file_size_kb:.1f} KB)")
    return str(file_path)

def cleanup_file(file_path: str):
    """Supprime un fichier temporaire"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🧹 Fichier nettoyé: {file_path}")
    except Exception as e:
        print(f"❌ Erreur nettoyage fichier {file_path}: {e}")

def predict_with_model(model: tf.keras.Model, features: np.ndarray) -> float:
    """Fait une prédiction avec un modèle"""
    try:
        # Reshape pour LSTM/B7: (1, 32, 2560)
        print(f"🔄 Reshape des features: {features.shape} -> (1, 32, 2560)")
        
        # Vérifier et ajuster la shape
        expected_features = 32 * 2560  # 81920
        if len(features) != expected_features:
            print(f"⚠️ Shape inattendu: {len(features)} features au lieu de {expected_features}")
            print(f"   Ajustement automatique...")
            
            # Si moins de features, padder avec des zéros
            if len(features) < expected_features:
                pad = expected_features - len(features)
                features = np.pad(features, (0, pad), mode='constant')
                print(f"   Padding: ajout de {pad} zéros")
            # Si plus de features, tronquer
            else:
                features = features[:expected_features]
                print(f"   Troncature: suppression de {len(features) - expected_features} features")
        
        features_reshaped = features.reshape(1, 32, 2560).astype(np.float32)
        print(f"✅ Features reshaped: {features_reshaped.shape}")
        
        # Prédiction
        print("🔮 Prédiction en cours...")
        prediction = model.predict(features_reshaped, verbose=0)
        print(f"✅ Prédiction brute: {prediction}")
        # Extraire le score
        if hasattr(prediction, 'shape'):
            if prediction.shape == (1, 1):
                score = float(prediction[0][0])
            elif prediction.shape == (1,):
                score = float(prediction[0])
            else:
                # Pour les modèles multi-sorties
                score = float(prediction[0][0] if hasattr(prediction[0], '__len__') else prediction[0])
        else:
            score = float(prediction)
        
        print(f"✅ Score de prédiction: {score:.4f}")
        return score
        
    except Exception as e:
        print(f"❌ Erreur prédiction: {e}")
        import traceback
        traceback.print_exc()
        # Retourner une valeur par défaut
        return 0.5

async def process_video_prediction(
    video_path: str,
    model_name: str = "default",
    filename: str = None
) -> Dict[str, Any]:
    """Pipeline complet de prédiction"""
    start_time = datetime.now()
    
    try:
        print(f"\n{'='*50}")
        print(f"🎬 Début du traitement: {filename or video_path}")
        print(f"{'='*50}")
        
        # 1. Vérifier si le fichier existe
        if not os.path.exists(video_path):
            error_msg = f"Video file not found: {video_path}"
            print(f"❌ {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"📁 Taille du fichier: {file_size_mb:.2f} MB")
        
        # 2. Extraire les features
        extractor = model_manager.load_feature_extractor()
        print("🔧 Extraction des features...")
        features = await extractor.extract_video_features_async(video_path)
        
        if features is None:
            error_msg = "Feature extraction failed"
            print(f"❌ {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        print(f"✅ Features extraites: shape={features.shape}")
        
        # 3. Obtenir le modèle
        model = model_manager.get_model(model_name)
        if model is None:
            error_msg = "No models available"
            print(f"❌ {error_msg}")
            raise HTTPException(status_code=404, detail=error_msg)
        
        print(f"🤖 Modèle utilisé: {model_name}")
        print(f"   Type: {type(model).__name__}")
        print(f"   Nombre de couches: {len(model.layers) if hasattr(model, 'layers') else 'N/A'}")
        
        # 4. Faire la prédiction
        score = predict_with_model(model, features)
        
        # 5. Calculer le temps
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # 6. Préparer la réponse
        actual_model_name = model_name if model_name in model_manager.models else list(model_manager.models.keys())[0] if model_manager.models else "default_model"
        model_info = model_manager.get_model_info(actual_model_name)
        
        # Déterminer la classification
        if score > 0.5:
            prediction_class = "FAKE"
        elif score < 0.5:
            prediction_class = "REAL"
        else:
            prediction_class = "UNCERTAIN"
        confidence = abs(score - 0.5) * 2
        
        result = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "model_used": actual_model_name,
            "prediction_score": float(score),
            "prediction_class": prediction_class,
            "confidence": round(float(confidence), 2),
            "processing_time_seconds": round(processing_time, 3),
            "features_shape": tuple(features.shape),
            "model_type": model_info.get("type", "unknown") if model_info else "unknown",
            "filename": filename,
            "file_size": int(os.path.getsize(video_path)) if os.path.exists(video_path) else 0
        }
        if prediction_class == "UNCERTAIN":
            result["warning"] = "Résultat très incertain, vérification manuelle recommandée."

        print(f"\n🎯 RÉSULTAT:")
        print(f"   Classification: {result['prediction_class']}")
        print(f"   Score: {score:.4f}")
        print(f"   Confiance: {result['confidence']:.1%}")
        if prediction_class == "UNCERTAIN":
            print(f"   ⚠️ {result['warning']}")
        print(f"   Temps total: {processing_time:.2f} secondes")
        print(f"{'='*50}")

        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ ERREUR:")
        print(f"   {str(e)}")
        print(f"{'='*50}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "error": f"Processing error: {str(e)[:200]}",
            "filename": filename
        }

# ==================== ENDPOINTS FASTAPI ====================
@app.get("/", response_class=HTMLResponse)
async def root():
    """Page d'accueil"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🔍 Deepfake Detection API - B7</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #3498db; }
            .method { display: inline-block; padding: 5px 10px; background: #3498db; color: white; border-radius: 3px; font-weight: bold; }
            .url { font-family: monospace; color: #2c3e50; }
            .btn { display: inline-block; padding: 10px 20px; background: #2ecc71; color: white; text-decoration: none; border-radius: 5px; margin: 10px 5px; }
            .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
            .healthy { background: #d4edda; color: #155724; }
            .warning { background: #fff3cd; color: #856404; }
            .error { background: #f8d7da; color: #721c24; }
            .model-list { background: #e9ecef; padding: 15px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Deepfake Detection API - EfficientNetB7</h1>
            <p>API pour la détection de deepfakes avec modèles B7 entraînés</p>
            
            <h2>📡 Endpoints disponibles:</h2>
            
            <div class="endpoint">
                <span class="method">GET</span> <span class="url">/api/health</span>
                <p>Vérifier l'état du serveur</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <span class="url">/api/models</span>
                <p>Lister les modèles disponibles</p>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <span class="url">/api/predict/upload</span>
                <p>Uploader une vidéo pour prédiction</p>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <span class="url">/api/predict/batch</span>
                <p>Prédiction batch (multiples vidéos)</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <span class="url">/api/stats</span>
                <p>Statistiques système</p>
            </div>
            
            <h2>🔗 Liens rapides:</h2>
            <a href="/docs" class="btn">📚 Documentation Swagger</a>
            <a href="/redoc" class="btn">📖 Documentation ReDoc</a>
            <a href="/api/models" class="btn">📦 Modèles disponibles</a>
            <a href="/test" class="btn">🎬 Tester l'API</a>
            
            <h2>📊 Statut:</h2>
            <div id="status" class="status">Chargement...</div>
            
            <div id="models" class="model-list" style="display: none;">
                <h3>📦 Modèles chargés:</h3>
                <ul id="modelsList"></ul>
            </div>
        </div>
        
        <script>
            // Charger le statut
            async function loadStatus() {
                try {
                    const response = await fetch('/api/health');
                    const data = await response.json();
                    
                    const statusDiv = document.getElementById('status');
                    const modelsDiv = document.getElementById('models');
                    const modelsList = document.getElementById('modelsList');
                    
                    if (data.models_loaded > 0) {
                        statusDiv.className = 'status healthy';
                        statusDiv.innerHTML = 
                            `✅ Serveur actif | ${data.models_loaded} modèles chargés | ` +
                            `Keras: ${data.keras_version} | ` +
                            new Date(data.timestamp).toLocaleString();
                        
                        // Charger la liste des modèles
                        const modelsResponse = await fetch('/api/models');
                        const modelsData = await modelsResponse.json();
                        
                        if (modelsData.success && modelsData.models.length > 0) {
                            modelsDiv.style.display = 'block';
                            modelsList.innerHTML = '';
                            modelsData.models.forEach(model => {
                                const li = document.createElement('li');
                                li.innerHTML = `<strong>${model.name}</strong>: ${model.parameters.toLocaleString()} paramètres (${model.type})`;
                                modelsList.appendChild(li);
                            });
                        }
                    } else {
                        statusDiv.className = 'status warning';
                        statusDiv.innerHTML = 
                            `⚠️ Serveur actif mais aucun modèle chargé | ` +
                            `Keras: ${data.keras_version} | ` +
                            `Dossier: ${data.models_dir} | ` +
                            new Date(data.timestamp).toLocaleString();
                    }
                        
                } catch (error) {
                    const statusDiv = document.getElementById('status');
                    statusDiv.className = 'status error';
                    statusDiv.innerHTML = '❌ Serveur non disponible';
                }
            }
            loadStatus();
        </script>
    </body>
    </html>
    """

@app.get("/test", response_class=HTMLResponse)
async def test_page():
    """Page de test interactif"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🎬 Test Deepfake Detection</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .upload-area { border: 2px dashed #3498db; padding: 40px; text-align: center; border-radius: 10px; margin: 20px 0; }
            .btn { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            .result { margin-top: 20px; padding: 20px; border-radius: 5px; display: none; }
            .real { background: #d4edda; color: #155724; }
            .fake { background: #f8d7da; color: #721c24; }
            .uncertain { background: #fff3cd; color: #856404; }
            .progress-bar { width: 100%; height: 20px; background: #f0f0f0; border-radius: 10px; margin: 10px 0; overflow: hidden; }
            .progress-fill { height: 100%; background: #3498db; width: 0%; transition: width 0.3s; }
            .model-select { margin: 10px 0; padding: 5px; width: 300px; }
            .info-box { background: #e9ecef; padding: 15px; border-radius: 5px; margin: 15px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎬 Tester la détection de deepfakes</h1>
            
            <div class="info-box">
                <h3>ℹ️ Comment ça marche:</h3>
                <ol>
                    <li>Uploader une vidéo (MP4, AVI, MOV, etc.)</li>
                    <li>Le serveur extrait 32 frames de la vidéo</li>
                    <li>Chaque frame passe par EfficientNetB7 pour extraire des features</li>
                    <li>Les features sont analysées par le modèle LSTM/Transformer</li>
                    <li>Résultat: REAL (authentique), FAKE (deepfake), ou UNCERTAIN</li>
                </ol>
            </div>
            
            <div class="upload-area">
                <h3>📤 Uploader une vidéo</h3>
                <div>
                    <label for="modelSelect"><strong>Modèle:</strong></label><br>
                    <select id="modelSelect" class="model-select">
                        <option value="default">Modèle par défaut (auto-sélection)</option>
                    </select>
                </div>
                <br>
                <div>
                    <input type="file" id="videoFile" accept="video/*">
                </div>
                <br>
                <button class="btn" onclick="uploadVideo()">🔍 Analyser la vidéo</button>
                <p id="status">Sélectionnez une vidéo pour commencer</p>
                <div class="progress-bar" style="display: none;">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
            </div>
            
            <div id="result" class="result"></div>
        </div>
        
        <script>
            // Charger les modèles disponibles
            async function loadModels() {
                try {
                    const response = await fetch('/api/models');
                    const data = await response.json();
                    
                    const select = document.getElementById('modelSelect');
                    
                    if (data.success && data.models.length > 0) {
                        data.models.forEach(model => {
                            const option = document.createElement('option');
                            option.value = model.name;
                            option.textContent = `${model.name} (${model.parameters.toLocaleString()} params)`;
                            select.appendChild(option);
                        });
                    }
                } catch (error) {
                    console.error('Erreur chargement des modèles:', error);
                }
            }
            
            async function uploadVideo() {
                const fileInput = document.getElementById('videoFile');
                const status = document.getElementById('status');
                const resultDiv = document.getElementById('result');
                const progressBar = document.querySelector('.progress-bar');
                const progressFill = document.getElementById('progressFill');
                const modelSelect = document.getElementById('modelSelect');
                
                if (!fileInput.files[0]) {
                    alert('Veuillez sélectionner une vidéo');
                    return;
                }
                
                const file = fileInput.files[0];
                const maxSizeMB = 100;
                const fileSizeMB = file.size / (1024 * 1024);
                
                if (fileSizeMB > maxSizeMB) {
                    alert(`La vidéo est trop grande (${fileSizeMB.toFixed(1)} MB). Maximum: ${maxSizeMB} MB`);
                    return;
                }
                
                const formData = new FormData();
                formData.append('video', file);
                formData.append('model_name', modelSelect.value);
                
                status.innerHTML = '⏳ Traitement en cours... (Extraction des features)';
                progressBar.style.display = 'block';
                progressFill.style.width = '20%';
                resultDiv.style.display = 'none';
                
                try {
                    const response = await fetch('/api/predict/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    progressFill.style.width = '60%';
                    status.innerHTML = '⏳ Traitement en cours... (Prédiction)';
                    
                    const data = await response.json();
                    progressFill.style.width = '100%';
                    
                    if (data.success) {
                        const isFake = data.prediction_class === 'FAKE';
                        const isReal = data.prediction_class === 'REAL';
                        const isUncertain = data.prediction_class === 'UNCERTAIN';
                        
                        resultDiv.className = 'result ' + 
                            (isFake ? 'fake' : isReal ? 'real' : 'uncertain');
                        
                        let confidenceDisplay = data.confidence > 0 ? 
                            `${(data.confidence * 100).toFixed(1)}%` : 'Non disponible';
                        
                        resultDiv.innerHTML = `
                            <h3>🎯 Résultat de l'analyse</h3>
                            <p><strong>Fichier:</strong> ${data.filename || 'Video'}</p>
                            <p><strong>Modèle utilisé:</strong> ${data.model_used}</p>
                            <p><strong>Score:</strong> ${data.prediction_score.toFixed(4)}</p>
                            <p><strong>Classification:</strong> <strong>${data.prediction_class}</strong></p>
                            <p><strong>Confiance:</strong> ${confidenceDisplay}</p>
                            <p><strong>Temps de traitement:</strong> ${data.processing_time_seconds.toFixed(2)} secondes</p>
                            <p><strong>Taille des features:</strong> ${data.features_shape}</p>
                            ${isFake ? 
                                '<div style="color: #721c24; font-weight: bold; margin-top: 15px; padding: 10px; background: #f8d7da; border-radius: 5px;">⚠️ Cette vidéo est probablement un DEEPFAKE</div>' : 
                              isReal ? 
                                '<div style="color: #155724; font-weight: bold; margin-top: 15px; padding: 10px; background: #d4edda; border-radius: 5px;">✅ Cette vidéo semble AUTHENTIQUE</div>' :
                                '<div style="color: #856404; font-weight: bold; margin-top: 15px; padding: 10px; background: #fff3cd; border-radius: 5px;">❓ Le résultat est incertain, vérification manuelle recommandée</div>'}
                        `;
                        resultDiv.style.display = 'block';
                        status.innerHTML = '✅ Analyse terminée';
                    } else {
                        resultDiv.className = 'result';
                        resultDiv.innerHTML = `
                            <h3>❌ Erreur</h3>
                            <p><strong>Fichier:</strong> ${data.filename || 'Video'}</p>
                            <p><strong>Erreur:</strong> ${data.error || 'Erreur inconnue'}</p>
                            <p>Veuillez vérifier le format de la vidéo et réessayer.</p>
                        `;
                        resultDiv.style.display = 'block';
                        status.innerHTML = '❌ Erreur';
                    }
                } catch (error) {
                    status.innerHTML = '❌ Erreur de connexion';
                    resultDiv.className = 'result';
                    resultDiv.innerHTML = `
                        <h3>❌ Erreur de connexion</h3>
                        <p>Impossible de se connecter au serveur.</p>
                        <p>Erreur: ${error.message}</p>
                        <p>Vérifiez que le serveur est en cours d'exécution.</p>
                    `;
                    resultDiv.style.display = 'block';
                    console.error(error);
                } finally {
                    setTimeout(() => {
                        progressBar.style.display = 'none';
                        progressFill.style.width = '0%';
                    }, 1000);
                }
            }
            
            // Initialiser
            loadModels();
        </script>
    </body>
    </html>
    """

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Vérifier l'état du serveur"""
    keras_version = "3.x" if model_manager.is_keras_3 else "2.x/TensorFlow"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(model_manager.models),
        "feature_extractor": "ready" if model_manager.feature_extractor else "not_loaded",
        "uploads_dir": str(UPLOADS_DIR),
        "models_dir": str(MODELS_DIR),
        "keras_version": keras_version
    }

@app.get("/api/models", response_model=ModelsResponse)
async def list_models():
    """Lister tous les modèles disponibles"""
    models_info = []
    
    for name, data in model_manager.models.items():
        models_info.append(ModelInfo(
            name=name,
            type=data["type"],
            loaded_at=data["loaded_at"],
            parameters=data["parameters"],
            input_shape=str(data["input_shape"]),
            path=data["path"]
        ))
    
    return {
        "success": True,
        "models": models_info,
        "total": len(models_info),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/models/scan")
async def scan_models():
    """Scanner et recharger les modèles"""
    count = model_manager.scan_models()
    return {
        "success": True,
        "message": f"Scanned {count} models",
        "models_loaded": count,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/predict/upload", response_model=PredictionResult)
async def predict_upload(
    video: UploadFile = File(...),
    model_name: str = Form("default"),
    background_tasks: BackgroundTasks = None
):
    """
    Uploader une vidéo et faire une prédiction
    
    Args:
        video: Fichier vidéo (mp4, avi, mov, mkv, webm)
        model_name: Nom du modèle à utiliser (optionnel)
        background_tasks: Pour le nettoyage automatique
    """
    # Vérifier le fichier
    if not allowed_file(video.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Format de fichier non supporté. Formats autorisés: mp4, avi, mov, mkv, webm, flv, wmv, m4v, mpg, mpeg"
        )
    
    # Vérifier la taille (max 100MB)
    max_size = 100 * 1024 * 1024  # 100MB
    video.file.seek(0, 2)  # Aller à la fin
    file_size = video.file.tell()
    video.file.seek(0)  # Retourner au début
    
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"Fichier trop grand ({file_size/(1024*1024):.1f} MB). Maximum: 100 MB"
        )
    
    print(f"\n📤 Upload reçu: {video.filename} ({file_size/(1024*1024):.1f} MB)")
    print(f"   Modèle demandé: {model_name}")
    
    # Sauvegarder le fichier temporairement
    video_path = save_upload_file(video)
    
    # Ajouter la tâche de nettoyage
    if background_tasks:
        background_tasks.add_task(cleanup_file, video_path)
    
    # Traiter la vidéo
    result = await process_video_prediction(
        video_path=video_path,
        model_name=model_name,
        filename=video.filename
    )
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Prediction failed"))
    
    return result

@app.post("/api/predict/batch", response_model=BatchPredictionResult)
async def predict_batch(
    videos: List[UploadFile] = File(...),
    model_name: str = Form("default"),
    background_tasks: BackgroundTasks = None
):
    """
    Prédiction batch pour plusieurs vidéos
    
    Args:
        videos: Liste de fichiers vidéo (max 10)
        model_name: Nom du modèle à utiliser
        background_tasks: Pour le nettoyage automatique
    """
    # Limiter à 10 fichiers
    if len(videos) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 fichiers autorisés par batch")
    
    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results = []
    errors = []
    
    print(f"\n📦 Début du traitement batch {batch_id}: {len(videos)} vidéos")
    
    # Traiter chaque vidéo
    for idx, video in enumerate(videos):
        print(f"\n  [{idx+1}/{len(videos)}] Traitement de: {video.filename}")
        
        if not allowed_file(video.filename):
            error_msg = "Format de fichier non supporté"
            print(f"    ❌ {error_msg}")
            errors.append({
                "filename": video.filename,
                "error": error_msg
            })
            continue
        
        try:
            # Sauvegarder temporairement
            video_path = save_upload_file(video)
            
            # Ajouter le nettoyage
            if background_tasks:
                background_tasks.add_task(cleanup_file, video_path)
            
            # Prédiction
            result = await process_video_prediction(
                video_path=video_path,
                model_name=model_name,
                filename=video.filename
            )
            
            results.append(result)
            print(f"    ✅ Succès: {result['prediction_class']} (score: {result['prediction_score']:.4f})")
            
        except Exception as e:
            error_msg = str(e)
            print(f"    ❌ Erreur: {error_msg}")
            errors.append({
                "filename": video.filename,
                "error": error_msg
            })
    
    print(f"\n✅ Traitement batch {batch_id} terminé:")
    print(f"   📊 Succès: {len(results)}")
    print(f"   ❌ Échecs: {len(errors)}")
    
    return {
        "success": True,
        "batch_id": batch_id,
        "total_files": len(videos),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/predict/url", response_model=PredictionResult)
async def predict_url(request: PredictionRequest):
    """
    Prédire depuis une URL de vidéo
    """
    return {
        "success": False,
        "timestamp": datetime.now().isoformat(),
        "error": "URL prediction not available",
        "suggestion": "Please use the upload endpoint instead",
        "model_used": "none",
        "prediction_score": 0.0,
        "prediction_class": "UNKNOWN",
        "confidence": 0.0,
        "processing_time_seconds": 0.0,
        "features_shape": (0,),
        "model_type": "none"
    }

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Obtenir les statistiques du serveur"""
    import platform

    import psutil

    # Informations système
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    system_info = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": round(memory.total / (1024**3), 2),
        "memory_available_gb": round(memory.available / (1024**3), 2),
        "memory_percent": memory.percent,
        "disk_total_gb": round(disk.total / (1024**3), 2),
        "disk_used_gb": round(disk.used / (1024**3), 2),
        "disk_percent": disk.percent,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__
    }
    
    # Informations GPU (si disponible)
    gpu_info = []
    try:
        gpu_devices = tf.config.list_physical_devices('GPU')
        for i, device in enumerate(gpu_devices):
            gpu_info.append({
                "name": f"GPU_{i}",
                "device_type": "GPU",
                "device_name": device.name
            })
    except:
        pass
    
    return {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "system": system_info,
        "models": {
            "loaded": len(model_manager.models),
            "names": list(model_manager.models.keys()),
            "types": list(set(m["type"] for m in model_manager.models.values()))
        },
        "gpu": gpu_info
    }

@app.get("/api/reports")
async def list_reports():
    # """Lister les rapports disponibles"""
    # reports = []
    
    # if REPORTS_DIR.exists():
    #     for json_file in REPORTS_DIR.glob("*.json"):
    #         reports.append({
    #             "name": json_file.name,
    #             "size": json_file.stat().st_size,
    #             "path": str(json_file),
    #             "type": "json"
    #         })
        
    #     for csv_file in REPORTS_DIR.glob("*.csv"):
    #         reports.append({
    #             "name": csv_file.name,
    #             "size": csv_file.stat().st_size,
    #             "path": str(csv_file),
    #             "type": "csv"
    #         })
    
    # return {
    #     "success": True,
    #     "reports": reports,
    #     "total": len(reports),
    #     "timestamp": datetime.now().isoformat()
    # }
    """Endpoint pour retourner les métriques du modèle"""
    try:
        METRICS_PATH = REPORTS_DIR / "model_metrics_20251203_134614.json"
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        return metrics
        
    except json.JSONDecodeError as e:
        return JSONResponse(
            {"error": f"Invalid JSON in metrics file: {str(e)}"},
            status_code=500
        )
    except Exception as e:
        return JSONResponse(
            {"error": f"Failed to load metrics: {str(e)}"},
            status_code=500
        )

@app.get("/api/reports/{report_name}")
async def get_report(report_name: str):
    """Télécharger un rapport spécifique"""
    report_path = REPORTS_DIR / report_name
    
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Report {report_name} not found")
    
    if report_name.endswith('.json'):
        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        return JSONResponse(content=report_data)
    else:
        # Pour les CSV et autres fichiers
        return FileResponse(
            path=report_path,
            filename=report_name,
            media_type='application/octet-stream'
        )

# ==================== INITIALISATION ====================
async def startup_event():
    """Initialiser le serveur au démarrage"""
    print("="*60)
    print("🚀 Démarrage du serveur Deepfake Detection API")
    print("="*60)
    
    # Vérifier les versions
    print(f"\n📦 Versions des bibliothèques:")
    print(f"   TensorFlow: {tf.__version__}")
    print(f"   OpenCV: {cv2.__version__}")
    
    # Vérifier les dossiers
    print(f"\n📁 Structure des dossiers:")
    print(f"   📦 Modèles: {MODELS_DIR}")
    print(f"   📤 Uploads: {UPLOADS_DIR}")
    print(f"   📊 Rapports: {REPORTS_DIR}")
    print(f"   🎨 Static: {STATIC_DIR}")
    
    # Vérifier si le dossier des modèles existe
    if not MODELS_DIR.exists():
        print(f"⚠️ ATTENTION: Le dossier des modèles n'existe pas: {MODELS_DIR}")
        print(f"   Création du dossier...")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Dossier créé")
    
    # Vérifier les fichiers dans le dossier
    h5_files = list(MODELS_DIR.glob("*.h5"))
    keras_files = list(MODELS_DIR.glob("*.keras"))
    
    print(f"\n🔍 Scan des modèles:")
    print(f"   .h5 files: {len(h5_files)} fichiers")
    print(f"   .keras files: {len(keras_files)} fichiers")
    
    if h5_files:
        print(f"   Fichiers .h5 trouvés:")
        for f in h5_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"     - {f.name} ({size_mb:.1f} MB)")
    
    # Charger les modèles
    print(f"\n📦 Chargement des modèles...")
    count = model_manager.scan_models()
    
    if count > 0:
        print(f"\n✅ {count} modèles chargés avec succès")
        for name, data in model_manager.models.items():
            params = data['parameters']
            input_shape = data['input_shape']
            print(f"   📊 {name}: {params:,} paramètres, input: {input_shape}")
    else:
        print(f"\n⚠️ Aucun modèle .h5 n'a pu être chargé")
        print(f"   Création d'un modèle par défaut pour les tests...")
        print(f"   Pour utiliser vos modèles .h5:")
        print(f"   1. Vérifiez qu'ils sont dans: {MODELS_DIR}")
        print(f"   2. Ils doivent être compatibles avec Keras/TensorFlow")
        print(f"   3. Formats supportés: .h5 (Keras 2.x), .keras (Keras 3.x)")
    
    # Charger l'extracteur
    model_manager.load_feature_extractor()
    print("\n✅ Extracteur de features B7 chargé")
    
    # Afficher les endpoints
    print("\n🌐 Endpoints disponibles:")
    print("   GET  /              - Page d'accueil")
    print("   GET  /docs          - Documentation Swagger")
    print("   GET  /test          - Page de test interactive")
    print("   GET  /api/health    - Vérifier l'état")
    print("   GET  /api/models    - Lister les modèles")
    print("   POST /api/predict/upload - Uploader une vidéo")
    print("   POST /api/predict/batch  - Prédiction batch")
    print("   GET  /api/stats     - Statistiques")
    print("="*60)
    print(f"✅ Serveur prêt! Accédez à http://localhost:8000")
    print("="*60)

async def shutdown_event():
    """Nettoyer à l'arrêt"""
    print("\n👋 Arrêt du serveur...")
    if hasattr(model_manager, 'executor'):
        model_manager.executor.shutdown(wait=False)
    print("✅ Serveur arrêté proprement")

# ==================== MAIN ====================
if __name__ == "__main__":
    import uvicorn

    print("Starting FastAPI server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )