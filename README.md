## 🔊 EchoSense: An End-to-End Urban Sound Classifier

Live Demo: [Link to your Streamlit App will go here]

This is a machine learning project that classifies 50 different urban sounds from the ESC-50 dataset. The project demonstrates a full end-to-end MLOps workflow, from model experimentation in a Jupyter notebook to a containerized API and an interactive web UI.

## Project Architecture

This application consists of two main components:

1. **FastAPI Backend:** A containerized API that serves a trained EfficientNetB0 model. It receives an audio file and returns a JSON response with the top 3 predictions.

2. **Streamlit Frontend:** A user-friendly web interface that allows users to upload an audio file and see the classification results from the API.

## Model Performance

* The final model is a fine-tuned EfficientNetB0 (a state-of-the-art CNN) that was trained on mel-spectrograms of audio clips.

* Dataset: ESC-50 (50 Classes)

* Final Test Accuracy: 68.75%

* This was achieved after multiple experiments, including building a custom CNN from scratch (66.25% accuracy) and testing various data augmentation and fine-tuning strategies.

How to Run This Project Locally

1. Run the Backend API

The backend is containerized with Docker.

### 1. Build the Docker image
docker build -t echosense_api:latest .

### 2. Run the Docker container
docker run --rm -p 8000:8000 echosense_api:latest


The API will be available at http://127.0.0.1:8000.

2. Run the Frontend UI

The frontend is a Streamlit app.

### 1. (In a new terminal) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate # or .\venv\Scripts\activate

### 2. Install Python dependencies
pip install -r requirements.txt
pip install streamlit

### 3. Run the Streamlit app
streamlit run app.py


The UI will be available at http://localhost:8501.