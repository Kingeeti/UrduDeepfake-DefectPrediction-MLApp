
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import joblib
import os
from tensorflow.keras.models import load_model

# ========== CONFIG ==========
st.set_page_config(page_title="WHIS Classifier", layout="centered")

# ========== AUDIO FEATURE EXTRACTOR ==========
def extract_mfcc(audio_file, sr=16000, n_mfcc=13, max_len=200):
    y, _ = librosa.load(audio_file, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.flatten().reshape(1, -1)

# ========== LOAD MODEL FUNCTION ==========
@st.cache_resource
def load_pickle_model(path):
    return joblib.load(path)

@st.cache_resource
def load_keras_model(path):
    return load_model(path)

# ========== MODEL FILE OPTIONS ==========
BINARY_MODELS = {
    "Logistic Regression": "binary-models/logistic_regression_model.pkl",
    "SVM": "binary-models/svm_model.pkl",
    "Perceptron": "binary-models/perceptron_model.pkl",
    "DNN (Keras)": "binary-models/dnn_model.h5",
}

MULTILABEL_MODELS = {
    "Tuned Logistic Regression": "multi-label-classifier/logistic_regression_model.pkl",
    "Tuned SVM": "multi-label-classifier/svm_model.pkl",
    "Tuned DNN (Keras)": "multi-label-classifier/dnn_model.h5",
    "Tuned Perceptron (Batch)": "multi-label-classifier/perceptron_batch_mode.pkl",
    "Tuned Perceptron (Online)": "multi-label-classifier/perceptron_online_mode.pkl",
    
}

# ========== SIDEBAR ==========
task = st.sidebar.selectbox("Select Task", ["🎙️ Deepfake Audio Detection", "🐞 Software Defect Prediction"])

# ========== TASK 1: AUDIO ==========
if task == "🎙️ Deepfake Audio Detection":
    st.title("🎙️ Deepfake Audio Detection")
    model_name = st.sidebar.selectbox("Choose Model", list(BINARY_MODELS.keys()))
    model_path = BINARY_MODELS[model_name]
    audio_file = st.file_uploader("Upload a WAV file", type=["wav"])

    if audio_file:
        st.audio(audio_file)
        features = extract_mfcc(audio_file)

        if st.button("Run Prediction"):
            if model_name == "DNN (Keras)":
                model = load_keras_model(model_path)
                prob = float(model.predict(features)[0][0])
                pred = int(prob >= 0.5)
            else:
                model = load_pickle_model(model_path)
                if hasattr(model, "predict_proba"):
                    prob = float(model.predict_proba(features)[0][1])
                elif hasattr(model, "decision_function"):
                    raw_score = model.decision_function(features)
                    prob = float(1 / (1 + np.exp(-raw_score[0])))
                else:
                    prob = 0.5  # fallback
                pred = model.predict(features)[0]

            label = "Deepfake" if pred else "Bonafide"
            st.success(f"Prediction: **{label}**")
            st.metric("Confidence", f"{prob:.2000000000f}")

# ========== TASK 2: MULTI-LABEL ==========
elif task == "🐞 Software Defect Prediction":
    st.title("🐞 Software Defect Multi-Label Prediction")
    model_name = st.sidebar.selectbox("Choose Model", list(MULTILABEL_MODELS.keys()))
    model_path = MULTILABEL_MODELS[model_name]
    input_csv = st.file_uploader("Upload a feature vector (CSV with 1 row)", type=["csv"])

    if input_csv:
        input_df = pd.read_csv(input_csv)
        features = input_df.values

        if st.button("Run Prediction"):
            if model_name == "DNN (Keras)":
                model = load_keras_model(model_path)
                probs = model.predict(features)[0]
                preds = (probs >= 0.5).astype(int)
            else:
                model = load_pickle_model(model_path)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(features)[0]
                else:
                    probs = [None] * features.shape[1]
                preds = model.predict(features)[0]

            st.write("### Prediction Results")
            labels = input_df.columns.tolist() if len(input_df.columns) == len(preds) else [f"Label {i+1}" for i in range(len(preds))]
            for i, label in enumerate(labels):
                status = "✅" if preds[i] else "❌"
                conf = f" (confidence: {probs[i]:.2f})" if probs[i] is not None else ""
                st.write(f"- {label}: {status}{conf}")
