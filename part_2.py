import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import hamming_loss, f1_score, make_scorer, precision_score
from sklearn.pipeline import Pipeline
import warnings
import joblib
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Optional: suppress warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("dataset.csv")
df.drop(columns=['type_task'], inplace=True)

# Split features and labels
X_text = df['report']
Y = df.drop(columns=['report'])

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf.fit_transform(X_text)

# Train/Validation/Test split
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.1765, random_state=42)

# Logistic Regression
lr_model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
lr_model.fit(X_train, Y_train)
Y_val_pred_lr = lr_model.predict(X_val)

# SVM
svm_model = OneVsRestClassifier(LinearSVC())
svm_model.fit(X_train, Y_train)
Y_val_pred_svm = svm_model.predict(X_val)

# Perceptron (Batch)
perc_model = OneVsRestClassifier(Perceptron(max_iter=1000, eta0=1.0, random_state=42))
perc_model.fit(X_train, Y_train)
Y_val_pred_perc = perc_model.predict(X_val)

# Perceptron (Online Learning)
classes = [np.array([0, 1])] * Y_train.shape[1]
online_model = OneVsRestClassifier(Perceptron(max_iter=1, warm_start=True, eta0=1.0))

for i in range(X_train.shape[0]):
    X_sample = X_train[i]
    Y_sample = Y_train.iloc[i:i+1]
    if i == 0:
        online_model.estimators_ = []
        for j in range(Y_train.shape[1]):
            est = Perceptron(max_iter=1, warm_start=True)
            est.partial_fit(X_sample, Y_sample.iloc[:, j], classes=[0, 1])
            online_model.estimators_.append(est)
    else:
        for j, est in enumerate(online_model.estimators_):
            est.partial_fit(X_sample, Y_sample.iloc[:, j], classes=[0, 1])
Y_val_pred_online = np.array([est.predict(X_val) for est in online_model.estimators_]).T

# Hyperparameter tuning for Logistic Regression
f1_micro_scorer = make_scorer(f1_score, average='micro')

lr_pipeline = Pipeline([('clf', OneVsRestClassifier(LogisticRegression(max_iter=1000)))])
lr_params = {
    'clf_estimator_C': [0.1, 1, 10],
    'clf_estimator_penalty': ['l2']
}
lr_grid = GridSearchCV(lr_pipeline, param_grid=lr_params, scoring=f1_micro_scorer, cv=3, verbose=1)
lr_grid.fit(X_train, Y_train)
best_lr_model = lr_grid.best_estimator_

# Hyperparameter tuning for SVM
svm_pipeline = Pipeline([('clf', OneVsRestClassifier(LinearSVC()))])
svm_params = {
    'clf_estimator_C': [0.1, 1, 10],
    'clf_estimator_loss': ['hinge', 'squared_hinge']
}
svm_grid = GridSearchCV(svm_pipeline, param_grid=svm_params, scoring=f1_micro_scorer, cv=3, verbose=1)
svm_grid.fit(X_train, Y_train)
best_svm_model = svm_grid.best_estimator_

# Hyperparameter tuning for Batch Perceptron
perc_pipeline = Pipeline([('clf', OneVsRestClassifier(Perceptron(max_iter=1000, random_state=42)))])
perc_params = {
    'clf_estimator_eta0': [0.01, 0.1, 1],
    'clf_estimator_penalty': [None, 'l2', 'l1', 'elasticnet']
}
perc_grid = GridSearchCV(perc_pipeline, param_grid=perc_params, scoring=f1_micro_scorer, cv=3, verbose=1)
perc_grid.fit(X_train, Y_train)
best_perc_model = perc_grid.best_estimator_

# Hyperparameter tuning for Online Perceptron
online_pipeline = Pipeline([('clf', OneVsRestClassifier(Perceptron(max_iter=1, warm_start=True, eta0=1.0)))])
online_params = {
    'clf_estimator_eta0': [0.01, 0.1, 1],
    'clf_estimator_penalty': [None, 'l2', 'l1', 'elasticnet']
}
online_grid = GridSearchCV(online_pipeline, param_grid=online_params, scoring=f1_micro_scorer, cv=3, verbose=1)
online_grid.fit(X_train, Y_train)
best_online_model = online_grid.best_estimator_

# Deep Neural Network
try:
    X_train_dense = X_train.toarray()
    X_val_dense = X_val.toarray()

    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train_dense.shape[1],), kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        Dense(Y_train.shape[1], activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(X_train_dense, Y_train, epochs=20, batch_size=32, validation_data=(X_val_dense, Y_val), callbacks=[early_stopping], verbose=1)
    Y_val_pred_dnn = (model.predict(X_val_dense) > 0.5).astype(int)

    dnn_metrics = {
        "Hamming Loss": hamming_loss(Y_val, Y_val_pred_dnn),
        "Micro-F1": f1_score(Y_val, Y_val_pred_dnn, average='micro'),
        "Macro-F1": f1_score(Y_val, Y_val_pred_dnn, average='macro'),
        "Precision@k": precision_score(Y_val, Y_val_pred_dnn, average='micro', zero_division=0)
    }
except ImportError:
    dnn_metrics = {"Hamming Loss": None, "Micro-F1": None, "Macro-F1": None}
    print("TensorFlow not installed. Skipping DNN model.")

# Ensure the folder exists
output_folder = "multi-label-classifier"
os.makedirs(output_folder, exist_ok=True)

# Save models
def save_model(model, filename):
    try:
        joblib.dump(model, filename)
        print(f"Model saved to {filename}")
    except Exception as e:
        print(f"Failed to save model {filename}: {e}")

save_model(best_lr_model, os.path.join(output_folder, "logistic_regression_model.pkl"))
save_model(best_svm_model, os.path.join(output_folder, "svm_model.pkl"))
save_model(best_perc_model, os.path.join(output_folder, "perceptron_batch_model.pkl"))
save_model(best_online_model, os.path.join(output_folder, "perceptron_online_model.pkl"))

if 'model' in locals():
    model.save(os.path.join(output_folder, "dnn_model.h5"))
    print(f"Tuned DNN model saved to {os.path.join(output_folder, 'dnn_model.h5')}")

# Compile results
results = {
    "Model": ["Logistic Regression", "SVM", "Perceptron (Batch)", "Perceptron (Online)", "DNN"],
    "Hamming Loss": [
        hamming_loss(Y_val, Y_val_pred_lr),
        hamming_loss(Y_val, Y_val_pred_svm),
        hamming_loss(Y_val, Y_val_pred_perc),
        hamming_loss(Y_val, Y_val_pred_online),
        dnn_metrics["Hamming Loss"]
    ],
    "Micro-F1": [
        f1_score(Y_val, Y_val_pred_lr, average='micro'),
        f1_score(Y_val, Y_val_pred_svm, average='micro'),
        f1_score(Y_val, Y_val_pred_perc, average='micro'),
        f1_score(Y_val, Y_val_pred_online, average='micro'),
        dnn_metrics["Micro-F1"]
    ],
    "Macro-F1": [
        f1_score(Y_val, Y_val_pred_lr, average='macro'),
        f1_score(Y_val, Y_val_pred_svm, average='macro'),
        f1_score(Y_val, Y_val_pred_perc, average='macro'),
        f1_score(Y_val, Y_val_pred_online, average='macro'),
        dnn_metrics["Macro-F1"]
    ],
    "Performance@k": [
        precision_score(Y_val, Y_val_pred_lr, average='micro', zero_division=0),
        precision_score(Y_val, Y_val_pred_svm, average='micro', zero_division=0),
        precision_score(Y_val, Y_val_pred_perc, average='micro', zero_division=0),
        precision_score(Y_val, Y_val_pred_online, average='micro', zero_division=0),
        dnn_metrics["Precision@k"]
    ]
}

results_df = pd.DataFrame(results)
print("\n=== Model Performance Summary ===")
print(results_df)