"""
Module endpoints
----------------
Définit les routes de l'API pour l'inférence, l'upload de données, etc.
"""

import json
import logging
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import h5py
import librosa
import numpy as np
import tensorflow as tf
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.models.architecture import create_dfdc_model_final

# Configuration des logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Constantes
MODEL_H5 = Path('models/first/best_model_improved.h5')
METRICS_PATH = Path('models/first/model_metrics_20251203_134614.json')
MAX_FRAMES = 32
IMG_SIZE = (224, 224)
N_MFCC = 13
MAX_PAD_LEN = 64

# Global model variable
MODEL = None

# Modèles Pydantic pour la validation
class TopKItem(BaseModel):
    label: str
    prob: float

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    topK: List[TopKItem]
    model: str
    inference_ms: int
    timestamp: str
    filename: str

class ModelInfoResponse(BaseModel):
    loaded: bool
    model_file: str
    exists: bool
    metrics_file: str
    metrics_exists: bool
    model_type: Optional[str] = None
    num_layers: Optional[int] = None
    inputs: Optional[List[dict]] = None
    outputs: Optional[List[dict]] = None
    last_layer: Optional[dict] = None
    timestamp: str

# Patch LSTM pour compatibilité
@tf.keras.utils.register_keras_serializable(package="Custom")
class CompatibleLSTM(tf.keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        kwargs.pop("implementation", None)
        super().__init__(*args, **kwargs)

@tf.keras.utils.register_keras_serializable(package="Custom")
class CompatibleGRU(tf.keras.layers.GRU):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        kwargs.pop("implementation", None)
        super().__init__(*args, **kwargs)

def load_model():
    """Load the model with full LSTM compatibility
    
    WARNING: enable_unsafe_deserialization is enabled for compatibility.
    Only load trusted models.
    """
    global MODEL
    if MODEL is not None:
        return MODEL
    
    try:
        # Autorise le chargement de Lambda layer (modèles de confiance uniquement !)
        tf.keras.config.enable_unsafe_deserialization()
        
        if not MODEL_H5.exists():
            logger.error(f"Model file not found: {MODEL_H5}")
            return None
        
        try:
            with h5py.File(MODEL_H5, 'r') as hf:
                has_model_config = 'model_config' in hf
                if not has_model_config:
                    logger.info("Loading model weights only")
                    MODEL = create_dfdc_model_final(frames_per_video=MAX_FRAMES)
                    MODEL.load_weights(MODEL_H5)
                    return MODEL
        except Exception as e:
            logger.warning(f"H5 file check failed: {e}")
        
        custom_objects = {
            "LSTM": CompatibleLSTM,
            "CompatibleLSTM": CompatibleLSTM,
            "GRU": CompatibleGRU,
            "CompatibleGRU": CompatibleGRU,
        }
        
        MODEL = tf.keras.models.load_model(
            MODEL_H5,
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False
        )
        
        if MODEL is not None:
            logger.info("✅ Model fully loaded!")
            
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}", exc_info=True)
        MODEL = None
    
    return MODEL

@router.get("/health")
def health():
    """Simple health check endpoint."""
    return {"status": "ok"}

def extract_frames_with_debug(video_path: str, frames_limit: int = MAX_FRAMES, img_size: tuple = IMG_SIZE):
    """Extract frames with detailed debugging info."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    logger.info(f"📹 Video properties - FPS: {fps}, Total frames: {total_frames}, Duration: {duration:.2f}s")
    
    # Calculate frame interval if we need to sample
    if total_frames > frames_limit:
        frame_interval = max(1, total_frames // frames_limit)
        logger.info(f"📊 Sampling every {frame_interval} frames")
    else:
        frame_interval = 1
    
    while frame_count < frames_limit:
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"⚠️ Could not read frame {frame_count}")
            break
        
        # Sample frames if needed
        if total_frames > frames_limit:
            for _ in range(frame_interval - 1):
                cap.read()
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, img_size)
        frames.append(frame_resized)
        
        # Log first frame statistics
        if frame_count == 0:
            logger.info(f"📸 Frame 0 - shape: {frame_resized.shape}, "
                       f"dtype: {frame_resized.dtype}, "
                       f"min: {frame_resized.min()}, "
                       f"max: {frame_resized.max()}")
        
        frame_count += 1
    
    cap.release()
    
    if not frames:
        raise ValueError("❌ No frames extracted from video")
    
    frames_array = np.array(frames, dtype=np.float32)
    logger.info(f"✅ Extracted {len(frames)} frames, array shape: {frames_array.shape}")
    
    # Padding if needed
    if frames_array.shape[0] < frames_limit:
        logger.info(f"📦 Padding from {frames_array.shape[0]} to {frames_limit} frames")
        pad_shape = (frames_limit - frames_array.shape[0], 
                    img_size[0], img_size[1], 3)
        pad = np.zeros(pad_shape, dtype=np.float32)
        frames_array = np.concatenate([frames_array, pad], axis=0)
    
    # Normalize to [0, 1]
    frames_array = frames_array / 255.0
    
    logger.info(f"🔢 After normalization - min: {frames_array.min():.6f}, "
               f"max: {frames_array.max():.6f}, "
               f"mean: {frames_array.mean():.6f}, "
               f"std: {frames_array.std():.6f}")
    
    # Check for NaN or Inf values
    if np.any(np.isnan(frames_array)):
        logger.error("❌ NaN values detected in frames!")
    if np.any(np.isinf(frames_array)):
        logger.error("❌ Infinite values detected in frames!")
    
    return frames_array[:frames_limit]

def extract_audio_for_submission(video_path: str, n_mfcc: int = N_MFCC, max_pad_len: int = MAX_PAD_LEN):
    """Extract audio features from video with debugging."""
    audio_path = video_path + '.wav'
    
    try:
        # Extraction audio avec ffmpeg
        logger.info(f"🎵 Extracting audio from {video_path}")
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path,
            '-loglevel', 'error'
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            logger.warning(f"⚠️ FFmpeg warning: {result.stderr.decode()}")
        
        if not os.path.exists(audio_path):
            logger.warning("⚠️ Audio file not created, using zeros")
            return np.zeros((n_mfcc, max_pad_len))
        
        # Chargement et extraction MFCC
        y, sr = librosa.load(audio_path, sr=16000)
        logger.info(f"🎶 Audio loaded - duration: {len(y)/sr:.2f}s, sample rate: {sr}")
        
        # Check if audio is not silent
        if np.max(np.abs(y)) < 0.01:
            logger.warning("⚠️ Audio appears to be very quiet or silent")
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
        logger.info(f"🎚️ MFCC extracted - shape: {mfcc.shape}")
        
        # Padding/truncation
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            logger.info(f"📏 Padded MFCC to shape: {mfcc.shape}")
        else:
            mfcc = mfcc[:, :max_pad_len]
            logger.info(f"✂️ Truncated MFCC to shape: {mfcc.shape}")
        
        # Normalize MFCC
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ FFmpeg audio extraction failed: {e.stderr.decode()}")
        mfcc = np.zeros((n_mfcc, max_pad_len))
    except Exception as e:
        logger.error(f"❌ Audio processing failed: {e}", exc_info=True)
        mfcc = np.zeros((n_mfcc, max_pad_len))
    finally:
        # Nettoyage du fichier audio temporaire
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete audio temp file: {e}")
    
    logger.info(f"✅ MFCC final - shape: {mfcc.shape}, "
               f"min: {mfcc.min():.4f}, max: {mfcc.max():.4f}, "
               f"mean: {mfcc.mean():.4f}")
    
    return mfcc

def inspect_model_architecture():
    """Inspect model architecture for debugging."""
    if MODEL is None:
        return "Model not loaded"
    
    info = {
        "num_layers": len(MODEL.layers),
        "layers": []
    }
    
    logger.info("🔍 Model Architecture Inspection:")
    logger.info(f"   Number of layers: {len(MODEL.layers)}")
    
    # Check last layer
    if len(MODEL.layers) > 0:
        last_layer = MODEL.layers[-1]
        logger.info(f"   Last layer: {last_layer.name}, Type: {type(last_layer).__name__}")
        
        if hasattr(last_layer, 'activation'):
            logger.info(f"   Last layer activation: {last_layer.activation}")
    
    return info

def safe_get_numpy(obj):
    """Safely convert object to numpy array."""
    if isinstance(obj, np.ndarray):
        return obj
    elif hasattr(obj, 'numpy'):
        return obj.numpy()
    elif hasattr(obj, '__array__'):
        return np.array(obj)
    else:
        return obj

def get_shape_list(shape_obj):
    """Safely convert tensor shape to list."""
    if hasattr(shape_obj, 'as_list'):
        return shape_obj.as_list()
    elif hasattr(shape_obj, '__len__'):
        try:
            return list(shape_obj)
        except:
            return str(shape_obj)
    else:
        return str(shape_obj)

@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Endpoint for prediction with enhanced debugging."""
    
    global MODEL
    if MODEL is None:
        MODEL = load_model()
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs or model file."
        )
    
    start = time.time()
    video_path = None
    
    try:
        filename = file.filename
        logger.info(f"🚀 Starting prediction for file: {filename}")
        
        # Validation du type de fichier
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{file_ext}'. Allowed: {allowed_extensions}"
            )
        
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            video_path = tmp.name
            file_size = len(content) / (1024 * 1024)  # MB
            logger.info(f"💾 Saved temp file: {video_path}, Size: {file_size:.2f} MB")
        
        # --- DEBUG: Model inspection ---
        model_info = inspect_model_architecture()
        
        # --- Video preprocessing ---
        logger.info("🎥 Extracting video frames...")
        face_frames = extract_frames_with_debug(video_path)
        
        # Prepare inputs
        spatial_temporal = np.expand_dims(face_frames, axis=0)
        temporal_input = np.expand_dims(face_frames, axis=0)
        
        logger.info(f"📦 Model input shapes - "
                   f"spatial: {spatial_temporal.shape}, "
                   f"temporal: {temporal_input.shape}")
        
        # --- Audio preprocessing ---
        logger.info("🎵 Extracting audio features...")
        audio_features = extract_audio_for_submission(video_path)
        audio_features = np.expand_dims(audio_features, axis=0)
        audio_features = np.transpose(audio_features, (0, 2, 1))
        
        logger.info(f"🎶 Audio features shape: {audio_features.shape}, "
                   f"min: {audio_features.min():.6f}, "
                   f"max: {audio_features.max():.6f}, "
                   f"mean: {audio_features.mean():.6f}")
        
        # --- Test prediction with random data first ---
        logger.info("🧪 Testing model with random data...")
        test_random_frames = np.random.rand(1, MAX_FRAMES, IMG_SIZE[0], IMG_SIZE[1], 3).astype(np.float32)
        test_random_audio = np.random.rand(1, MAX_PAD_LEN, N_MFCC).astype(np.float32)
        
        try:
            test_pred = MODEL.predict([test_random_frames, test_random_frames, test_random_audio], 
                                     verbose=0)
            logger.info(f"🧪 Random test prediction type: {type(test_pred)}")
            
            # Convert to numpy safely
            test_pred_np = safe_get_numpy(test_pred)
            
            if isinstance(test_pred_np, list):
                for i, p in enumerate(test_pred_np):
                    p_array = safe_get_numpy(p)
                    logger.info(f"🧪 Output {i}: min={p_array.min():.6f}, max={p_array.max():.6f}, mean={p_array.mean():.6f}")
            else:
                logger.info(f"🧪 Test output: min={test_pred_np.min():.6f}, max={test_pred_np.max():.6f}, mean={test_pred_np.mean():.6f}")
        except Exception as e:
            logger.warning(f"⚠️ Random test failed: {e}")
        
        # --- Real prediction ---
        logger.info("🤖 Running real prediction...")
        
        # Get raw model outputs
        try:
            predictions = MODEL.predict([spatial_temporal, temporal_input, audio_features], 
                                       verbose=0)
            
            # Debug output structure
            logger.info(f"📊 Prediction structure type: {type(predictions)}")
            
            # Convert to numpy safely
            predictions_np = safe_get_numpy(predictions)
            
            if isinstance(predictions_np, list):
                logger.info(f"📊 Model has {len(predictions_np)} outputs")
                
                # Log all outputs for debugging
                for i, pred in enumerate(predictions_np):
                    pred_array = safe_get_numpy(pred)
                    
                    logger.info(f"📊 Output {i} shape: {pred_array.shape}")
                    logger.info(f"📊 Output {i} values (first 5): {pred_array.flatten()[:5]}")
                    logger.info(f"📊 Output {i} stats - min: {pred_array.min():.6f}, "
                               f"max: {pred_array.max():.6f}, mean: {pred_array.mean():.6f}")
                
                # Use main output (usually first)
                if len(predictions_np) > 0:
                    main_pred_array = safe_get_numpy(predictions_np[0])
                else:
                    main_pred_array = safe_get_numpy(predictions_np)
            else:
                main_pred_array = predictions_np
                
                logger.info(f"📊 Single output shape: {main_pred_array.shape}")
                logger.info(f"📊 Single output values: {main_pred_array.flatten()}")
                logger.info(f"📊 Single output stats - min: {main_pred_array.min():.6f}, "
                           f"max: {main_pred_array.max():.6f}, mean: {main_pred_array.mean():.6f}")
                
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}", exc_info=True)
            raise
        
        # Handle different output formats
        logger.info(f"📊 Raw prediction array shape: {main_pred_array.shape}")
        logger.info(f"📊 Raw prediction array: {main_pred_array}")
        
        # Extract prediction value
        if main_pred_array.ndim > 1:
            main_pred = float(main_pred_array[0][0])
        else:
            main_pred = float(main_pred_array[0])
        
        logger.info(f"🎯 Raw prediction value: {main_pred}")
        
        # Check if prediction needs activation function
        if main_pred > 10 or main_pred < -10:
            logger.warning(f"⚠️ Prediction outside typical range [-10,10]: {main_pred}")
            # Apply sigmoid to bring to [0,1]
            main_pred = 1 / (1 + np.exp(-main_pred))
            logger.info(f"🎯 After sigmoid: {main_pred}")
        elif main_pred < 0 or main_pred > 1:
            logger.warning(f"⚠️ Prediction outside [0,1] range: {main_pred}")
            # Apply sigmoid anyway
            main_pred = 1 / (1 + np.exp(-main_pred))
            logger.info(f"🎯 After sigmoid: {main_pred}")
        
        # Apply temperature scaling to avoid extreme confidence
        temperature = 2.0  # Higher temperature = softer probabilities
        if main_pred > 0.5:
            # Fake probability
            main_pred_temp = np.exp(np.log(main_pred + 1e-10) / temperature)
            real_pred_temp = np.exp(np.log(1 - main_pred + 1e-10) / temperature)
            main_pred = main_pred_temp / (main_pred_temp + real_pred_temp)
        else:
            # Real probability
            real_prob = 1 - main_pred
            real_pred_temp = np.exp(np.log(real_prob + 1e-10) / temperature)
            fake_pred_temp = np.exp(np.log(main_pred + 1e-10) / temperature)
            main_pred = fake_pred_temp / (fake_pred_temp + real_pred_temp)
        
        logger.info(f"🎯 After temperature scaling (T={temperature}): {main_pred}")
        
        # Clip to avoid extreme values
        main_pred = np.clip(main_pred, 0.05, 0.95)
        logger.info(f"🎯 After clipping: {main_pred}")
        
        # --- Create response ---
        label = "Fake" if main_pred > 0.5 else "Real"
        confidence = round(float(main_pred), 4)
        real_prob = round(1 - confidence, 4)
        
        # Ensure probabilities sum to 1
        if abs(confidence + real_prob - 1.0) > 0.001:
            total = confidence + real_prob
            confidence = round(confidence / total, 4)
            real_prob = round(real_prob / total, 4)
        
        topK = [
            {"label": "Fake", "prob": confidence},
            {"label": "Real", "prob": real_prob},
        ]
        
        inference_ms = int((time.time() - start) * 1000)
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        logger.info(f"✅ Final result - label: {label}, confidence: {confidence}, "
                   f"inference time: {inference_ms}ms")
        
        return PredictionResponse(
            label=label,
            confidence=confidence,
            topK=topK,
            model="Improved DFDC Deepfake Detector",
            inference_ms=inference_ms,
            timestamp=timestamp,
            filename=filename
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"❌ ValueError: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
    finally:
        # Nettoyage du fichier temporaire
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                logger.info(f"🗑️ Cleaned up temp file: {video_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete temp file {video_path}: {e}")

@router.post("/test_random")
async def test_random():
    """Test the model with random input to check output range."""
    global MODEL
    if MODEL is None:
        MODEL = load_model()
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded."
        )
    
    try:
        # Create random inputs
        random_frames = np.random.rand(1, MAX_FRAMES, IMG_SIZE[0], IMG_SIZE[1], 3).astype(np.float32)
        random_audio = np.random.rand(1, MAX_PAD_LEN, N_MFCC).astype(np.float32)
        
        logger.info("🧪 Running random input test...")
        logger.info(f"🧪 Random frames shape: {random_frames.shape}")
        logger.info(f"🧪 Random audio shape: {random_audio.shape}")
        
        # Get prediction
        prediction = MODEL.predict([random_frames, random_frames, random_audio], verbose=0)
        
        response = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "test_type": "random_input"
        }
        
        # Convert to numpy safely
        prediction_np = safe_get_numpy(prediction)
        
        if isinstance(prediction_np, list):
            response["num_outputs"] = len(prediction_np)
            for i, pred in enumerate(prediction_np):
                pred_array = safe_get_numpy(pred)
                response[f"output_{i}"] = {
                    "shape": list(pred_array.shape),
                    "min": float(pred_array.min()),
                    "max": float(pred_array.max()),
                    "mean": float(pred_array.mean()),
                    "std": float(pred_array.std()),
                    "raw_first_5": pred_array.flatten()[:5].tolist()
                }
            
            # Get first output for further analysis
            if prediction_np:
                pred_flat = safe_get_numpy(prediction_np[0]).flatten()
            else:
                pred_flat = np.array([])
        else:
            response["single_output"] = {
                "shape": list(prediction_np.shape),
                "min": float(prediction_np.min()),
                "max": float(prediction_np.max()),
                "mean": float(prediction_np.mean()),
                "std": float(prediction_np.std()),
                "raw_values": prediction_np.flatten().tolist()
            }
            pred_flat = prediction_np.flatten()
        
        # Apply activation functions to see results
        if len(pred_flat) > 0:
            # Test sigmoid
            sigmoid_vals = 1 / (1 + np.exp(-pred_flat))
            response["sigmoid_applied"] = {
                "min": float(sigmoid_vals.min()),
                "max": float(sigmoid_vals.max()),
                "mean": float(sigmoid_vals.mean())
            }
            
            # Test softmax (for 2 classes)
            if len(pred_flat) >= 2:
                softmax_vals = np.exp(pred_flat[:2])
                softmax_vals = softmax_vals / softmax_vals.sum()
                response["softmax_applied"] = softmax_vals.tolist()
        
        logger.info(f"🧪 Test completed successfully")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Random test error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Test error: {str(e)}"
        )

@router.get("/model_info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model."""
    global MODEL
    if MODEL is None:
        MODEL = load_model()
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded."
        )
    
    try:
        info = {
            "loaded": MODEL is not None,
            "model_file": str(MODEL_H5),
            "exists": MODEL_H5.exists(),
            "metrics_file": str(METRICS_PATH),
            "metrics_exists": METRICS_PATH.exists(),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        if MODEL is not None:
            info["model_type"] = type(MODEL).__name__
            info["num_layers"] = len(MODEL.layers)
            
            # Input info
            info["inputs"] = []
            if hasattr(MODEL, 'inputs'):
                for i, inp in enumerate(MODEL.inputs):
                    shape_list = get_shape_list(inp.shape)
                    info["inputs"].append({
                        "index": i,
                        "name": inp.name if hasattr(inp, 'name') else f"input_{i}",
                        "shape": shape_list
                    })
            
            # Output info
            info["outputs"] = []
            if hasattr(MODEL, 'output'):
                if isinstance(MODEL.output, list):
                    for i, out in enumerate(MODEL.outputs):
                        shape_list = get_shape_list(out.shape)
                        info["outputs"].append({
                            "index": i,
                            "name": out.name if hasattr(out, 'name') else f"output_{i}",
                            "shape": shape_list
                        })
                else:
                    shape_list = get_shape_list(MODEL.output.shape)
                    info["outputs"].append({
                        "index": 0,
                        "name": MODEL.output.name if hasattr(MODEL.output, 'name') else "output",
                        "shape": shape_list
                    })
            
            # Last layer info
            if len(MODEL.layers) > 0:
                last_layer = MODEL.layers[-1]
                layer_config = {}
                try:
                    layer_config = last_layer.get_config()
                except:
                    layer_config = {"error": "Could not get config"}
                
                info["last_layer"] = {
                    "name": last_layer.name if hasattr(last_layer, 'name') else "unknown",
                    "type": type(last_layer).__name__,
                    "config": str(layer_config)
                }
        
        return ModelInfoResponse(**info)
        
    except Exception as e:
        logger.error(f"❌ Model info error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Model info error: {str(e)}"
        )

@router.get("/metrics")
def get_metrics():
    """Endpoint pour retourner les métriques du modèle"""
    try:
        if not METRICS_PATH.exists():
            return JSONResponse(
                {"error": f"Metrics file not found at {METRICS_PATH}"},
                status_code=404
            )
        
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        logger.info(f"✅ Loaded metrics from {METRICS_PATH}")
        return metrics
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in metrics file: {e}")
        return JSONResponse(
            {"error": f"Invalid JSON in metrics file: {str(e)}"},
            status_code=500
        )
    except Exception as e:
        logger.error(f"❌ Failed to load metrics: {e}")
        return JSONResponse(
            {"error": f"Failed to load metrics: {str(e)}"},
            status_code=500
        )

# Load model on startup
@router.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("🚀 Starting up API server...")
    model = load_model()
    if model:
        logger.info("✅ Model loaded successfully on startup")
    else:
        logger.error("❌ Failed to load model on startup")