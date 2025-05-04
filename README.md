# UrduDeepfake-DefectPrediction-MLApp
An end-to-end machine learning project that combines Urdu deepfake audio detection and multi-label software defect prediction, wrapped in a real-time Streamlit app.
Project Overview
This project contains:
 Part 1: Binary classification of Urdu audio (Bonafide vs Deepfake)
 Part 2: Multi-label classification of software defect data
 Part 3: Streamlit-based UI for real-time predictions

Features:
Deepfake Audio Detection:
Audio preprocessing with MFCCs and Spectrograms
Models: SVM, Logistic Regression, Perceptron, Deep Neural Network
Metrics: Accuracy, Precision, Recall, F1, AUC-ROC

Software Defect Prediction:
Preprocessing: Missing values, scaling, label imbalance
Models: One-vs-Rest Logistic Regression, Multi-label SVM, Online Perceptron, DNN
Metrics: Hamming Loss, Micro/Macro F1, Precision@k

Streamlit App:
Upload audio for deepfake prediction
Input feature vector for defect label prediction
Choose model (SVM, Logistic Regression, DNN) at runtime
View confidence scores

Getting Started
bash
Copy
Edit
# Clone the repo
git clone https://github.com/yourusername/UrduDeepfake-DefectPrediction-MLApp.git
cd UrduDeepfake-DefectPrediction-MLApp

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
Dataset Sources:
Urdu Deepfake Dataset: CSALT/deepfake_detection_dataset_urdu

Software Defect Dataset: Provided in CSV format

Technologies Used:
Python, Scikit-learn, TensorFlow/Keras
Librosa, Pandas, NumPy
Streamlit, Matplotlib

License:
This project is for educational and research purposes.

