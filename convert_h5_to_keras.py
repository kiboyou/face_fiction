import os

from tensorflow import keras

# Chemin vers ton ancien modèle H5
H5_PATH = "models/first/best_optimized_model.h5"
# Chemin de sortie du nouveau modèle Keras v3
KERAS_PATH = "models/first/model_1764769178.keras"

if not os.path.isfile(H5_PATH):
    raise FileNotFoundError(f"Model file does not exist: {H5_PATH}")

model = keras.models.load_model(H5_PATH, compile=False)
model.save(KERAS_PATH)
print(f"✔️ Model converti et sauvegardé sous : {KERAS_PATH}")
