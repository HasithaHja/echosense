import streamlit as st
import pandas as pd
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import EfficientNetB0
import pickle
import io

# Constants & Model Paths
TARGET_SIZE = (224, 224)
MODEL_PATH = "model_assets/echosense_champion_model.keras"
ENCODER_PATH = "model_assets/echosense_label_encoder.pkl"


# Load Model & Encoder ONCE
@st.cache_resource
def load_echosense_model():
    """
    Loads the trained model and label encoder from disk.
    This function is cached to run only once.
    """
    try:
        # Load encoder
        with open(ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        NUM_CLASSES = len(label_encoder.classes_)

        # Build model architecture
        def build_model(num_classes):
            input_shape = (TARGET_SIZE[0], TARGET_SIZE[1], 3)
            base_model = EfficientNetB0(input_shape=input_shape, include_top=False, weights=None)
            base_model.trainable = False 
            model = Sequential([
                base_model,
                GlobalAveragePooling2D(),
                Dropout(0.5),
                Dense(num_classes, activation='softmax')
            ])
            return model
        
        # Load weights
        model = build_model(NUM_CLASSES)
        model.load_weights(MODEL_PATH)
        
        print("Model and encoder loaded successfully.")
        return model, label_encoder
    
    except FileNotFoundError:
        print("Error: Model or encoder file not found.")
        st.error("**Error: Model or encoder file not found.** Please ensure `model_assets/echosense_champion_model.keras` and `model_assets/echosense_label_encoder.pkl` exist.")
        return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        st.error(f"**An error occurred while loading the model:** {e}")
        return None, None


# Preprocessing Function
def preprocess_audio(file_stream):
    """
    Loads an audio file stream, processes it, and returns a
    3-channel spectrogram ready for the model.
    """
    try:
        # Load audio data from the in-memory file stream
        audio_data, sr = librosa.load(file_stream, sr=22050, duration=5.0)
        
        # Ensure 5-second duration
        audio_data = librosa.util.fix_length(data=audio_data, size=22050 * 5)
        
        # Create the mel-spectrogram
        spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=22050)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        
        # Resize and convert to 3-channel
        spectrogram_resized = tf.image.resize(np.expand_dims(spectrogram, axis=-1), TARGET_SIZE)
        spectrogram_3channel = np.concatenate([spectrogram_resized, spectrogram_resized, spectrogram_resized], axis=-1)
        
        # Add batch dimension
        spectrogram_3channel = np.expand_dims(spectrogram_3channel, axis=0)
        
        return spectrogram_3channel

    except Exception as e:
        print(f"Error during audio processing: {e}")
        return None

# Page Configuration & UI
st.set_page_config(
    page_title="EchoSense: Urban Sound Classifier",
    page_icon="🔊",
    layout="wide"
)

# Load model and encoder
model, label_encoder = load_echosense_model()

# Sidebar
with st.sidebar:
    st.header("About EchoSense")
    st.markdown("""
    This demo is an all-in-one Streamlit app that classifies 50 different urban sounds.
    
    - **Model**: `EfficientNetB0`
    - **Dataset**: ESC-50
    - **Final Accuracy**: 68.75%
    - **Deployment**: **Hugging Face Spaces**
    """)
    st.markdown("---")
    st.markdown("""
    The full project, including the containerized **FastAPI backend** and **Dockerfile**, is available on GitHub.
    """)
    st.markdown("[View Project on GitHub](https://github.com/HasithaHja/echosense)")

# Main Page Content
st.title("🔊 EchoSense")
st.header("Urban Sound Classifier")

if model is None or label_encoder is None:
    st.error("**Error: Model files failed to load.** The app cannot start. Please check the logs.")
else:
    st.markdown("""
    Upload an audio file (e.g., `.wav`, `.mp3`) and the model will 
    classify what sound it is.
    """)

    uploaded_file = st.file_uploader(
        "Choose an audio file...", 
        type=["wav", "mp3", "m4a", "flac"]
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        with st.spinner('Analyzing the sound...'):
            # Call functions directly, no API request.
            processed_audio = preprocess_audio(uploaded_file)
            
            if processed_audio is None:
                st.error("Error: Could not process the audio file.")
            else:
                # Run prediction
                prediction_probs = model.predict(processed_audio, verbose=0)
                
                # Get Top 3 
                top_3_indices = np.argsort(prediction_probs[0])[-3:][::-1]
                
                result = {
                    "top_3_predictions": [
                        {"class": label_encoder.classes_[i], "confidence": float(prediction_probs[0][i])}
                        for i in top_3_indices
                    ]
                }
                result["predicted_class"] = result["top_3_predictions"][0]["class"]
                result["confidence"] = result["top_3_predictions"][0]["confidence"]
                # End of key change 

                # Display results
                top_class = result['predicted_class'].replace('_', ' ').title()
                confidence_pct = result['confidence'] * 100
                
                if confidence_pct > 75:
                    st.success(f"**Top Prediction: {top_class}**")
                elif confidence_pct > 40:
                    st.info(f"**Top Prediction: {top_class}**")
                else:
                    st.warning(f"**Top Prediction: {top_class}** (Low confidence)")

                st.metric(label="Confidence", value=f"{confidence_pct:.2f}%")
                
                st.subheader("Top 3 Predictions")
                df = pd.DataFrame(result['top_3_predictions'])
                df.columns = ["Predicted Class", "Confidence (%)"]
                df['Predicted Class'] = df['Predicted Class'].str.replace('_', ' ').str.title()
                df['Confidence (%)'] = df['Confidence (%)'].apply(lambda x: f"{x * 100:.2f}%")
                st.dataframe(df, use_container_width=True)

st.divider()
st.markdown("An end-to-end ML project.")

