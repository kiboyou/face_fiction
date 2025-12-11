
import os

from tensorflow import keras
from tensorflow.keras.layers import LSTM

print("Contenu du dossier models/first :", repr(os.listdir("models/first/best_optimized_model.weights.keras")))

# Teste d'abord le chemin absolu pour lever toute ambiguïté
MODEL_PATH = os.path.abspath("models/first/best_optimized_model.weights.keras")
print("Chemin absolu testé :", MODEL_PATH)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model file does not exist: {MODEL_PATH}")

model = keras.models.load_model(MODEL_PATH)
print("✔️ Model loaded")

# import os

# path = "models/first/best_optimized_model.weights.keras"
# print("Exists:", os.path.exists(path))
# print("Size:", os.path.getsize(path), "bytes")
