import os
import pickle
import librosa
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import EfficientNetB0

## Define the target size for our spectrograms
TARGET_SIZE = (224, 224)
MODEL_PATH = "model_assets/echosense_champion_model.keras"
ENCODER_PATH = "model_assets/echosense_label_encoder.pkl"

## Load the label encoder
try:
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print("Label encoder loaded successfully.")
    NUM_CLASSES = len(label_encoder.classes_)
except Exception as e:
    print(f"FATAL ERROR: Could not load label encoder. Error: {e}")
    label_encoder = None
    sys.exit(1)

## Build the model
def build_model(num_classes):
    """
    Creates the exact same model architecture we used in Colab.
    """
    input_shape = (TARGET_SIZE[0], TARGET_SIZE[1], 3)
    
    # Load the pre-trained base (weights=None because we'll load our own)
    base_model = EfficientNetB0(input_shape=input_shape, include_top=False, weights=None)
    base_model.trainable = False # Not needed for inference, but good practice

    # Build our new model on top
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

## Load the trained model
try:
    model = build_model(NUM_CLASSES)
    # Load only the weights from our saved file
    model.load_weights(MODEL_PATH) 
    print("Model architecture built and weights loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not build model or load weights. Error: {e}")
    model = None

## Fail fast
if model is None or label_encoder is None:
    print("One or more essential files failed to load. Exiting.")
    sys.exit(1)

## Initialize FastAPI app
app = FastAPI(title='EchoSense API')

## CORS for a frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

## Preprocessing function
def preprocess_audio(file_stream):
    """
    Loads an audio file stream, processes it, and returns a
    3-channel spectrogram ready for the model.
    """
    try:
        # Load audio data from the in-memory file stream
        # Handles multiple formats (wav, mp3, etc.) and resampling
        audio_data, sr = librosa.load(file_stream, sr=22050, duration=5.0)
        
        # Ensure 5-second duration (pad if shorter, truncate if longer)
        audio_data = librosa.util.fix_length(data=audio_data, size=22050 * 5)
        
        # 1. Create the mel-spectrogram
        spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=22050)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        
        # 2. Resize to the target size
        spectrogram_resized = tf.image.resize(np.expand_dims(spectrogram, axis=-1), TARGET_SIZE)
        
        # 3. Convert grayscale to 3-channel
        spectrogram_3channel = np.concatenate([spectrogram_resized, spectrogram_resized, spectrogram_resized], axis=-1)
        
        # 4. Add batch dimension
        spectrogram_3channel = np.expand_dims(spectrogram_3channel, axis=0)
        
        return spectrogram_3channel

    except Exception as e:
        print(f"Error during audio processing: {e}")
        return None


## ____________ API Endpoints ___________________________

@app.get("/")
def read_root():
    """A simple root endpoint to check if the API is running."""
    return {"message": "Welcome to the EchoSense API. Send a POST request to /predict to classify an audio file."}

## Health check
@app.get("/health")
def health_check():
    """Allows monitoring tools to check if the service is alive."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "encoder_loaded": label_encoder is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives an audio file, processes it, and returns the
    predicted sound category with its confidence score.
    """

    # --- FIX #4: Add file validation ---
    allowed_extensions = {'.wav', '.mp3', '.flac', '.m4a'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not supported. Please upload one of: {allowed_extensions}"
        )

    # Read the file content into an in-memory stream
    file_stream = io.BytesIO(await file.read())

    # Preprocess the audio
    processed_audio = preprocess_audio(file_stream)
    
    if processed_audio is None:
        raise HTTPException(status_code=400, detail="Could not process the uploaded audio file.")




    # Run prediction
    try:
        # Add verbose=0 to hide the prediction progress bar in the terminal
        prediction_probs = model.predict(processed_audio, verbose=0) 
        
        # Get top 3 predictions
        # [0] to get the predictions for the first (and only) item in the batch
        top_3_indices = np.argsort(prediction_probs[0])[-3:][::-1] # Get top 3 indices
        
        # Get the top prediction details
        predicted_index = top_3_indices[0]
        predicted_class = label_encoder.classes_[predicted_index]
        confidence = float(prediction_probs[0][predicted_index])
        
        # Create the list of top 3 results
        top_3_results = [
            {
                "class": label_encoder.classes_[idx],
                "confidence": float(prediction_probs[0][idx])
            }
            for idx in top_3_indices
        ]

        # Return the new response format
        return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_3_predictions": top_3_results 
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

# --- 5. Main entry point to run the app ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
