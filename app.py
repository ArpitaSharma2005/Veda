from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import json

# Initialize the FastAPI application
app = FastAPI()

# --- 1. Load your model, class names, AND plant details ---
MODEL_PATH = "plant_model_final.keras"
model = load_model(MODEL_PATH)

with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_map = {v: k for k, v in class_indices.items()}

# NEW: Load the plant details from the new JSON file
with open('plant_details.json', 'r') as f:
    plant_details = json.load(f)

print("✅ Model and all data loaded successfully!")


# --- 2. Preprocessing function (no change here) ---
def preprocess_image(img, target_size=(256, 256)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


# --- 3. Define API endpoints ---
@app.get("/")
def home():
    return {"message": "Plant Identification API is Running 🌿"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    processed_img = preprocess_image(img)

    preds = model.predict(processed_img)
    predicted_class_index = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))

    # Get the common name from your class map
    common_name = class_map.get(predicted_class_index, "Unknown Plant")
    
    # NEW: Get the scientific name from the plant details file
    details = plant_details.get(common_name, {})
    scientific_name = details.get('scientific_name', 'Not Found')

    # UPDATED: Return all details in the final result
    return {
        "common_name": common_name,
        "botanical_name": scientific_name,
        "confidence": f"{confidence * 100:.2f}%"
    }