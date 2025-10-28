# EchoSense: An End-to-End Urban Sound Classifier

**Live Application Demo:** [https://huggingface.co/spaces/HHaaraa/echosense](https://huggingface.co/spaces/HHaaraa/echosense)

This is a full-stack, end-to-end machine learning project that classifies environmental sounds from the ESC-50 dataset. It is deployed as a two-part microservice application: a containerized FastAPI backend and a Streamlit frontend UI, both hosted on Hugging Face Spaces.

- **Live Backend API:** [https://hhaaraa-echosense-api.hf.space/health](https://hhaaraa-echosense-api.hf.space/health)
- **Live Frontend UI:** [https://huggingface.co/spaces/HHaaraa/echosense](https://huggingface.co/spaces/HHaaraa/echosense)
- **Kaggle Notebook (My Research):** [Link to your Kaggle Notebook here]
---

## 1. Project Goal

The objective was to build and deploy a production-ready machine learning service capable of classifying 50 different urban sounds. This project demonstrates the complete MLOps lifecycle: from data exploration and model iteration in a notebook to building a robust, containerized API and deploying it with a live user interface.

---

## 2. Model & Performance

After multiple experiments, the champion model was a **Transfer Learning model using EfficientNetB0** (pre-trained on ImageNet).

- **Dataset:** ESC-50 (2000 audio clips, 50 classes)
- **Features:** Mel-spectrograms resized to (224, 224) and 3-channeled
- **Final Test Accuracy:** 68.75%

This model was chosen over a custom-built CNN (which achieved 66.25%) after extensive hyperparameter tuning. The full research and model comparison can be found in my [Kaggle Notebook]([Link to your Kaggle Notebook here]).

---

## 3. Architecture

This application is deployed as a two-part microservice system hosted on Hugging Face Spaces:

### Backend (`/backend`)
- **Framework:** FastAPI
- **Containerization:** Docker
- **Function:** Serves the trained TensorFlow/Keras model (`.keras` file) and the LabelEncoder (`.pkl` file). It exposes `/health` and `/predict` endpoints.
- **Deployment:** Hosted as a Hugging Face Docker Space.

### Frontend (`/frontend`)
- **Framework:** Streamlit
- **Function:** Provides a clean user interface for file uploads. It sends a request to the backend API and displays the Top-3 predictions with confidence scores.
- **Deployment:** Hosted as a Hugging Face Streamlit Space.

---

## 4. Challenges & Lessons Learned

A significant part of this project was navigating real-world deployment challenges on a free-tier CPU-only environment.

### TensorFlow/GPU vs. CPU
The model was trained on a GPU (Colab) but deployed to a CPU-only environment. This required forcing TensorFlow to run in CPU-only mode by setting `ENV CUDA_VISIBLE_DEVICES="-1"` in the Dockerfile.

### Deployment Caching
The `librosa` library (and its dependencies `joblib` and `numba`) aggressively caches data, which fails on a read-only filesystem.

**Solution:** The final solution was a "bulletproof" Dockerfile that explicitly creates and sets permissions for its own cache directory (`RUN mkdir /cache`, `RUN chmod 777 /cache`) and sets environment variables to force all caching to that one writable folder (`ENV JOBLIB_CACHE_DIR=/cache`, `ENV NUMBA_CACHE_DIR=/cache`).

---

## 5. How to Run Locally

You can run this entire two-part application on your local machine.

### 1. Run the Backend API
```bash
# Navigate to the 'backend' folder
cd backend

# Build the docker image
docker build -t echosense_api:latest .

# Run the container
docker run --rm -p 8000:8000 echosense_api:latest
```

### 2. Run the Frontend UI
```bash
# (In a new terminal)
# Navigate to the 'frontend' folder
cd frontend

# Create a virtual environment and install libraries
python -m venv venv
source venv/bin/activate  # (or .\venv\Scripts\activate on Windows)
pip install -r requirements.txt

# Run the streamlit app
streamlit run app.py
```

The app will be available at [http://localhost:8501](http://localhost:8501).

---

## Tech Stack

- **Machine Learning:** TensorFlow/Keras, EfficientNetB0, librosa
- **Backend:** FastAPI, uvicorn, Docker
- **Frontend:** Streamlit
- **Deployment:** Hugging Face Spaces
- **Data Processing:** NumPy, pickle