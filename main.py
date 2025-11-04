from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # 1. Import the middleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import pandas as pd
import io
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
# from tensorflow.keras.applications.imagenet_utils import preprocess_input # No longer needed

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration (Brain) ---
BRAIN_MODEL_PATH = "brain_disease_classifier_model.keras"
BRAIN_IMAGE_SIZE = (224, 224)
BRAIN_CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --- Configuration (Lung) ---
LUNG_MODEL_PATH = "lung_cancer_model.keras"
LUNG_IMAGE_SIZE = (224, 224)
# --- CORRECTED CLASS ORDER (0 = No Cancer, 1 = Cancer) ---
LUNG_CLASS_NAMES = ['No Cancer', 'Cancer'] # <-- ADJUST TO YOUR NEEDS, e.g., ['No Cancer', 'Cancer']

# --- Configuration (Heart) ---
HEART_MODEL_PATH = "ECG_model.keras"
SCALER_PATH = "ecg_scaler.joblib"
EXPECTED_TIMESTEPS = 187
HEART_CLASS_NAMES = [
    "Normal beat (N)", 
    "Supraventricular ectopic beat (S)", 
    "Ventricular ectopic beat (V)", 
    "Fusion beat (F)", 
    "Unknown beat (Q)"
]

# --- Load Models and Scaler ---
try:
    brain_model = tf.keras.models.load_model(BRAIN_MODEL_PATH)
    logger.info("Brain model loaded.")
except Exception as e:
    logger.error(f"Error loading brain model: {e}")
    brain_model = None

try:
    lung_model = tf.keras.models.load_model(LUNG_MODEL_PATH)
    logger.info("Lung model loaded.")
except Exception as e:
    logger.error(f"Error loading lung model: {e}")
    lung_model = None

try:
    heart_model = tf.keras.models.load_model(HEART_MODEL_PATH)
    logger.info("Heart model loaded.")
except Exception as e:
    logger.error(f"Error loading heart model: {e}")
    heart_model = None

try:
    scaler = joblib.load(SCALER_PATH)
    logger.info("Scaler loaded.")
except Exception as e:
    logger.error(f"Error loading scaler: {e}")
    scaler = None

# --- Preprocessing & Prediction Functions ---

# --- Brain Functions ---
def preprocess_brain_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(BRAIN_IMAGE_SIZE)
        image_array = np.array(image)
        image_batch = np.expand_dims(image_array, axis=0)
        image_batch_float = tf.cast(image_batch, tf.float32)
        
        # --- CORRECTED NORMALIZATION ---
        # Models are almost always trained on [0, 1] scaling
        processed_image = image_batch_float / 255.0 
        
        return processed_image.numpy()
    except Exception as e:
        logger.error(f"Error during brain image preprocessing: {e}")
        raise HTTPException(status_code=400, detail="Failed to preprocess image.")

def get_brain_prediction(processed_image: np.ndarray) -> dict:
    if brain_model is None:
        raise HTTPException(status_code=500, detail="Brain model is not loaded.")
    try:
        # This model is multi-class, so argmax is correct
        predictions = brain_model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])
        predicted_class_name = BRAIN_CLASS_NAMES[predicted_class_index]
        return {"predicted_class": predicted_class_name, "confidence": round(confidence, 4)}
    except Exception as e:
        logger.error(f"Error during brain prediction: {e}")
        raise HTTPException(status_code=500, detail="Failed to run brain prediction.")

# --- Lung Functions ---
def preprocess_lung_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(LUNG_IMAGE_SIZE)
        image_array = np.array(image)
        image_batch = np.expand_dims(image_array, axis=0)
        image_batch_float = tf.cast(image_batch, tf.float32)
        
        # --- CORRECTED NORMALIZATION ---
        processed_image = image_batch_float / 255.0 
        
        return processed_image.numpy()
    except Exception as e:
        logger.error(f"Error during lung image preprocessing: {e}")
        raise HTTPException(status_code=400, detail="Failed to preprocess image.")

def get_lung_prediction(processed_image: np.ndarray) -> dict:
    if lung_model is None:
        raise HTTPException(status_code=500, detail="Lung model is not loaded.")
    try:
        # --- CORRECTED BINARY LOGIC ---
        # predict() returns a single value (e.g., [0.1883])
        prediction_raw = lung_model.predict(processed_image)[0][0]
        
        # Use 0.5 as the threshold (as in your notebook)
        predicted_class_index = int(prediction_raw > 0.5) # 0 or 1
        
        # Calculate confidence based on the prediction
        if predicted_class_index == 1:
            confidence = float(prediction_raw)
        else:
            confidence = 1.0 - float(prediction_raw)
            
        predicted_class_name = LUNG_CLASS_NAMES[predicted_class_index]
        
        return {"predicted_class": predicted_class_name, "confidence": round(confidence, 4)}
    except Exception as e:
        logger.error(f"Error during lung prediction: {e}")
        raise HTTPException(status_code=500, detail="Failed to run lung prediction.")


# --- Heart Functions ---
def preprocess_ecg_data(csv_bytes: bytes) -> np.ndarray:
    if scaler is None:
        raise HTTPException(status_code=500, detail="Scaler is not loaded.")
    try:
        try:
            # Try reading as a single row
            s = str(csv_bytes, 'utf-8')
            data_io = io.StringIO(s)
            ecg_data = pd.read_csv(data_io, header=None).iloc[0].values 
        except Exception:
            # Try reading as a single column
            s = str(csv_bytes, 'utf-8')
            data_io = io.StringIO(s)
            ecg_data = pd.read_csv(data_io, header=None).iloc[:, 0].values

        if len(ecg_data) != EXPECTED_TIMESTEPS:
            raise ValueError(f"ECG data must have {EXPECTED_TIMESTEPS} points, got {len(ecg_data)}.")
            
        ecg_data = ecg_data.astype(np.float32).reshape(1, -1) 
        ecg_scaled = scaler.transform(ecg_data)
        ecg_reshaped = ecg_scaled.reshape(1, EXPECTED_TIMESTEPS, 1)
        return ecg_reshaped
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error during ECG preprocessing: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to preprocess CSV data: {e}")

def get_ecg_prediction(processed_ecg: np.ndarray) -> dict:
    if heart_model is None:
        raise HTTPException(status_code=500, detail="Heart model is not loaded.")
    try:
        # This model is multi-class, so argmax is correct
        predictions = heart_model.predict(processed_ecg)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])
        predicted_class_name = HEART_CLASS_NAMES[predicted_class_index]
        return {"predicted_class": predicted_class_name, "confidence": round(confidence, 4)}
    except Exception as e:
        logger.error(f"Error during ECG prediction: {e}")
        raise HTTPException(status_code=500, detail="Failed to run ECG prediction.")


# --- API Response Models ---
class BrainAnalysisResult(BaseModel):
    patientId: str
    scanType: str
    predicted_class: str
    confidence: float
    notes: str | None = None

class LungAnalysisResult(BaseModel):
    patientId: str
    scanType: str
    predicted_class: str
    confidence: float
    notes: str | None = None

class HeartAnalysisResult(BaseModel):
    patientId: str
    recordingDate: str
    predicted_class: str
    confidence: float
    notes: str | None = None

# --- API Application ---
app = FastAPI()

# Load environment variables from .env
load_dotenv()

# Get frontend URL from environment
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", FRONTEND_URL, "http://127.0.0.1:3000"],  # Dynamically allows your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 🏠 Landing Page Route ---
@app.get("/")
async def root():
    logger.info("Landing page accessed.")
    return {"message": "NeuroPulmoCardiaNet API is running."}

# --- Brain Route ---
@app.post("/api/analyze/brain", response_model=BrainAnalysisResult)
async def analyze_brain(
    image: UploadFile = File(...),
    patientId: str = Form(...),
    scanType: str = Form(...),
    notes: str = Form("")
):
    logger.info(f"Received brain analysis request for patient ID: {patientId}")
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image data received.")
    
    processed_image = preprocess_brain_image(image_bytes)
    prediction_results = get_brain_prediction(processed_image)
    
    result = BrainAnalysisResult(
        patientId=patientId,
        scanType=scanType,
        predicted_class=prediction_results.get("predicted_class"),
        confidence=prediction_results.get("confidence"),
        notes=notes if notes else None
    )
    return result

# --- Lung Route ---
@app.post("/api/analyze/lung", response_model=LungAnalysisResult)
async def analyze_lung(
    image: UploadFile = File(...),
    patientId: str = Form(...),
    scanType: str = Form(...), # e.g., "X-Ray", "CT Scan"
    notes: str = Form("")
):
    logger.info(f"Received lung analysis request for patient ID: {patientId}")
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image data received.")
    
    processed_image = preprocess_lung_image(image_bytes)
    prediction_results = get_lung_prediction(processed_image)
    
    result = LungAnalysisResult(
        patientId=patientId,
        scanType=scanType,
        predicted_class=prediction_results.get("predicted_class"),
        confidence=prediction_results.get("confidence"),
        notes=notes if notes else None
    )
    return result

# --- Heart Route ---
@app.post("/api/analyze/heart", response_model=HeartAnalysisResult)
async def analyze_heart(
    csvFile: UploadFile = File(...),
    patientId: str = Form(...),
    recordingDate: str = Form(...),
    notes: str = Form("")
):
    logger.info(f"Received heart analysis request for patient ID: {patientId}")
    csv_bytes = await csvFile.read()
    if not csv_bytes:
        raise HTTPException(status_code=400, detail="No CSV data received.")
    
    processed_ecg = preprocess_ecg_data(csv_bytes)
    prediction_results = get_ecg_prediction(processed_ecg)
    
    result = HeartAnalysisResult(
        patientId=patientId,
        recordingDate=recordingDate,
        predicted_class=prediction_results.get("predicted_class"),
        confidence=prediction_results.get("confidence"),
        notes=notes if notes else None
    )
    return result

# --- Run Server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)