import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Optional: suppress warnings
warnings.filterwarnings('ignore')

# === STEP 2: DEFINE FEATURE EXTRACTION FUNCTION ===
def extract_features_recursive(root_dir, label_value, sr=16000, n_mfcc=13, max_len=200):
    features, labels = [], []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".wav"):
                try:
                    file_path = os.path.join(dirpath, file)
                    signal, _ = librosa.load(file_path, sr=sr)
                    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
                    if mfcc.shape[1] < max_len:
                        pad_width = max_len - mfcc.shape[1]
                        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
                    else:
                        mfcc = mfcc[:, :max_len]
                    features.append(mfcc.flatten())
                    labels.append(label_value)
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")
    return features, labels

# === STEP 3: LOAD DATA FROM LOCAL FOLDERS ===
base_dir = r"./deepfake_detection_dataset_urdu"
bonafide_feat, bonafide_lbl = extract_features_recursive(os.path.join(base_dir, "Bonafide"), 0)
tacotron_feat, tacotron_lbl = extract_features_recursive(os.path.join(base_dir, "Spoofed_Tacotron"), 1)
vits_feat, vits_lbl = extract_features_recursive(os.path.join(base_dir, "Spoofed_TTS"), 1)

X = np.array(bonafide_feat + tacotron_feat + vits_feat)
y = np.array(bonafide_lbl + tacotron_lbl + vits_lbl)

print("Feature shape:", X.shape)
print("Label distribution:", np.bincount(y))

# === STEP 4: SPLIT AND SCALE ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === STEP 5: CLASSICAL MODELS ===
# Logistic Regression Hyperparameter Tuning with RandomizedSearchCV
lr_pipeline = Pipeline([
    ('clf', LogisticRegression(max_iter=1000))
])

lr_params = {
    'clf__C': [0.1, 1, 10],
    'clf__penalty': ['l2'],
    'clf__solver': ['liblinear', 'saga']
}

lr_random_search = RandomizedSearchCV(lr_pipeline, param_distributions=lr_params, n_iter=10, scoring='roc_auc', cv=3, verbose=1, random_state=42, n_jobs=-1)
lr_random_search.fit(X_train_scaled, y_train)
best_lr = lr_random_search.best_estimator_

# SVM Hyperparameter Tuning with RandomizedSearchCV
svm_pipeline = Pipeline([
    ('clf', SVC(probability=True))
])

svm_params = {
    'clf__C': [0.1, 1, 10],
    'clf__kernel': ['linear', 'rbf'],
    'clf__gamma': ['scale', 'auto']
}

svm_random_search = RandomizedSearchCV(svm_pipeline, param_distributions=svm_params, n_iter=10, scoring='roc_auc', cv=3, verbose=1, random_state=42, n_jobs=-1)
svm_random_search.fit(X_train_scaled, y_train)
best_svm = svm_random_search.best_estimator_

# Perceptron Hyperparameter Tuning with RandomizedSearchCV
perc_pipeline = Pipeline([
    ('clf', Perceptron(max_iter=1000))
])

perc_params = {
    'clf__eta0': [0.01, 0.1, 0.5, 1],
    'clf__penalty': [None, 'l2', 'l1', 'elasticnet']
}

perc_random_search = RandomizedSearchCV(perc_pipeline, param_distributions=perc_params, n_iter=10, scoring='roc_auc', cv=3, verbose=1, random_state=42, n_jobs=-1)
perc_random_search.fit(X_train_scaled, y_train)
best_perc = perc_random_search.best_estimator_

# Online Perceptron Hyperparameter Tuning with RandomizedSearchCV
online_pipeline = Pipeline([
    ('clf', Perceptron(max_iter=1, warm_start=True, eta0=1.0))
])

online_params = {
    'clf__eta0': [0.01, 0.1, 1],
    'clf__penalty': [None, 'l2', 'l1', 'elasticnet']
}

online_random_search = RandomizedSearchCV(online_pipeline, param_distributions=online_params, n_iter=10, scoring='roc_auc', cv=3, verbose=1, random_state=42, n_jobs=-1)
online_random_search.fit(X_train_scaled, y_train)
best_online_model = online_random_search.best_estimator_

# === STEP 6: DNN ===
dnn = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],), kernel_regularizer=l2(0.01)),
    Dropout(0.4),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

dnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

dnn.fit(
    X_train_scaled, y_train,
    epochs=20, batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

y_prob_dnn = dnn.predict(X_test_scaled).flatten()
y_pred_dnn = (y_prob_dnn > 0.5).astype(int)

# === STEP 7: SAVE MODELS ===
def save_model(model, filename):
    try:
        joblib.dump(model, filename)
        print(f"Model saved to {filename}")
    except Exception as e:
        print(f"Failed to save model {filename}: {e}")

output_dir = "binary-models"
os.makedirs(output_dir, exist_ok=True)

save_model(best_lr, os.path.join(output_dir, "logistic_regression_model.pkl"))
save_model(best_svm, os.path.join(output_dir, "svm_model.pkl"))
save_model(best_perc, os.path.join(output_dir, "perceptron_model.pkl"))
save_model(best_online_model, os.path.join(output_dir, "perceptron_online_model.pkl"))

try:
    dnn.save(os.path.join(output_dir, "dnn_model.h5"))
    print(f"Tuned DNN model saved to {os.path.join(output_dir, 'dnn_model.h5')}")
except Exception as e:
    print(f"Failed to save DNN model: {e}")

# === STEP 8: EVALUATION ===
def print_results(name, y_true, y_pred, y_prob=None):
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred))
    if y_prob is not None:
        print("AUC-ROC:", roc_auc_score(y_true, y_prob))

print_results("Tuned Logistic Regression", y_test, best_lr.predict(X_test_scaled), best_lr.predict_proba(X_test_scaled)[:, 1])
print_results("Tuned SVM", y_test, best_svm.predict(X_test_scaled), best_svm.predict_proba(X_test_scaled)[:, 1])
print_results("Tuned Perceptron", y_test, best_perc.predict(X_test_scaled))
print_results("Tuned Online Perceptron", y_test, best_online_model.predict(X_test_scaled))
print_results("Tuned DNN", y_test, y_pred_dnn, y_prob_dnn)