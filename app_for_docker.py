import streamlit as st
import requests
import pandas as pd

# Define the backend URL
API_URL = "http://127.0.0.1:8000/predict"
API_HEALTH_URL = "http://127.0.0.1:8000/health"

## Page configurations
st.set_page_config(
    page_title="EchoSense: Urban Sound Classifier",
    layout="centered"
)

## Sidebar
with st.sidebar:
    st.header("About EchoSense")
    st.markdown("""
    This app is an end-to-end audio classification project.
    
    - **Model**: `EfficientNetB0` (Transfer Learning)
    - **Dataset**: ESC-50 (50 sound classes)
    - **Final Accuracy**: 68.75%
    - **Architecture**: FastAPI backend serving a TensorFlow/Keras model.
    """)
    st.markdown("---")
    st.markdown("[View Project on GitHub](https://github.com/yourusername/echosense)")

## API health check
@st.cache_data(ttl=5) # Cache the check for 5 seconds
def check_api_health():
    """Checks if the backend API is running."""
    try:
        response = requests.get(API_HEALTH_URL, timeout=2)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

api_is_running = check_api_health()


## Content
st.title("🔊 EchoSense")
st.header("Urban Sound Classifier")
st.markdown("""
Upload an audio file (e.g., `.wav`, `.mp3`, `.m4a`) and the model will 
classify what sound it is.

This app is a frontend for a FastAPI backend that serves a 
TensorFlow/Keras model trained on the ESC-50 dataset.
""")

## File uploader
uploaded_file = st.file_uploader(
    "Choose an audio file...",
    type=["wav", "mp3", "m4a", "flac"]
)

if uploaded_file is not None:
    # Display the audio player
    st.audio(uploaded_file, format='audio/wav')

    # Display a spinner while processing
    with st.spinner('Analyzing the sound...'):
        try:
            # Create a dictionary for the uploaded files
            files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}

            # Send the file to the API
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                # Get the JSON response
                result = response.json()

                # Display the prediction
                top_class = result['predicted_class'].replace('_', ' ').title()
                confidence_pct = result['confidence'] * 100
                
                if confidence_pct > 75:
                    st.success(f"**Top Prediction: {top_class}**")
                elif confidence_pct > 40:
                    st.info(f"**Top Prediction: {top_class}**")
                else:
                    st.warning(f"**Top Prediction: {top_class}** (Low confidence)")

                st.metric(label="Confidence", value=f"{confidence_pct:.2f}%")
                
                # Display the top 3 predictions in a clean table
                st.subheader("Top 3 Predictions")
                df = pd.DataFrame(result['top_3_predictions'])
                df.columns = ["Predicted Class", "Confidence (%)"]
                df['Predicted Class'] = df['Predicted Class'].str.replace('_', ' ').str.title()
                df['Confidence (%)'] = df['Confidence (%)'].apply(lambda x: f"{x * 100:.2f}%")
                st.dataframe(df, use_container_width=True)

            else:
                # Show error details from the API
                st.error(f"Error from API (Status {response.status_code}): {response.json().get('detail', 'Unknown error')}")

        except requests.exceptions.ConnectionError:
            st.error("Connection Error: Could not connect to the API. Is the backend (Docker container) running?")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")


