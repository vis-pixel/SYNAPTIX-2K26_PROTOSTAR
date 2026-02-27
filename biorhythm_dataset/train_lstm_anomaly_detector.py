"""
================================================================================
 BioRhythm Fusion Band — LSTM Autoencoder for Anomaly Detection
 Train on Normal Baseline, Detect Micro-Deviations via Reconstruction Error
================================================================================

This script loads the pre-generated npz dataset (biorhythm_windows.npz),
filters the training set to only include "Normal" physiology (Label 0),
and trains an LSTM Autoencoder to learn the personalized baseline mapping of
the 7 fundamental signals (HR, HRV, SKT, EDA, SPO2, SMF, CRS).

Once trained, inputs containing early fever micro-deviations will produce a
LARGE reconstruction error (MSE), which translates directly to the Fever Risk Score.
"""

import os
import json
import numpy as np

# Try to use TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Input, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("\n[!] TensorFlow is not installed. To run the training, please: pip install tensorflow")


def load_data(filepath="d:/empty/biorhythm_dataset/biorhythm_windows.npz"):
    print(f"Loading dataset from: {filepath}")
    data = np.load(filepath)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    
    # Autoencoders should only train on NORMAL data to learn the baseline
    train_normal_idx = np.where(y_train == 0)[0]
    X_train_normal = X_train[train_normal_idx]
    
    val_normal_idx = np.where(y_val == 0)[0]
    X_val_normal = X_val[val_normal_idx]

    print(f"Original Train: {X_train.shape} -> Normal Only: {X_train_normal.shape}")
    print(f"Original Val:   {X_val.shape} -> Normal Only: {X_val_normal.shape}")
    
    return X_train_normal, X_val_normal, X_test, y_test, data["feature_names"]


def build_lstm_autoencoder(timesteps=60, features=7):
    """
    Architecture:
    Encoder: Condenses the 60-step, 7-channel series into a 16-dim latent vector.
    Decoder: Reconstructs the 60-step time series from the latent vector.
    """
    if not HAS_TF:
        return None

    model = Sequential([
        Input(shape=(timesteps, features)),
        
        # ── ENCODER ──
        LSTM(64, activation='tanh', return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='tanh', return_sequences=False),
        
        # Latent Space (Bottleneck)
        # Repeat the latent vector `timesteps` times for the decoder
        RepeatVector(timesteps),
        
        # ── DECODER ──
        LSTM(32, activation='tanh', return_sequences=True),
        Dropout(0.2),
        LSTM(64, activation='tanh', return_sequences=True),
        TimeDistributed(Dense(features))  # Reconstruct original 7 features
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model


def main():
    print("=" * 80)
    print("  BioRhythm Fusion Band LSTM Anomaly Detector")
    print("=" * 80)
    
    # 1. Load Data
    biorhythm_path = "d:/empty/biorhythm_dataset/biorhythm_windows.npz"
    if not os.path.exists(biorhythm_path):
        print(f"[!] Cannot find {biorhythm_path}. Please run generate_dataset.py first.")
        return
        
    X_train, X_val, X_test, y_test, features = load_data(biorhythm_path)
    
    if not HAS_TF:
        print("\nSkipping training because TensorFlow is missing.")
        print("Install via: pip install tensorflow")
        return
        
    timesteps = X_train.shape[1]
    num_features = X_train.shape[2]
    
    # 2. Build Model
    model = build_lstm_autoencoder(timesteps, num_features)
    model.summary()
    
    # 3. Train Model
    print("\n[>>>] Training LSTM Autoencoder on NORMAL physiological baseline...")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(filepath="d:/empty/biorhythm_dataset/biorhythm_lstm_model.h5", 
                        monitor='val_loss', save_best_only=True)
    ]
    
    # Using 10 epochs for demo purposes (usually 50+ in production)
    history = model.fit(
        X_train, X_train,
        epochs=10,
        batch_size=128,
        validation_data=(X_val, X_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # 4. Evaluation
    print("\n[>>>] Evaluating Model on Test Set (Mixed Normal & Fever Windows)...")
    
    # Reconstruct test set
    X_test_pred = model.predict(X_test, batch_size=256)
    
    # Mean Squared Error per window (Deviation Index)
    # Average error across all 60 timesteps and 7 features
    test_mse = np.mean(np.square(X_test - X_test_pred), axis=(1, 2))
    
    # 5. Calculate Risk Threshold
    # Determine what a "normal" reconstruction error is using the Validation set
    X_val_pred = model.predict(X_val, batch_size=256)
    val_mse = np.mean(np.square(X_val - X_val_pred), axis=(1, 2))
    
    # Threshold = Mean + 3 Std Devs of normal data
    threshold = np.mean(val_mse) + 3.0 * np.std(val_mse)
    print(f"\n[!] Calculated Normal Baseline Anomaly Threshold: {threshold:.4f}")
    
    # 6. Detect Anomalies in Test Set
    anomalies = test_mse > threshold
    
    true_anomalies = (y_test > 0)  # Labels 1-7 are various diseases
    
    # Accuracy metrics
    TP = np.sum(anomalies & true_anomalies)
    TN = np.sum(~anomalies & ~true_anomalies)
    FP = np.sum(anomalies & ~true_anomalies)
    FN = np.sum(~anomalies & true_anomalies)
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*40)
    print("  LSTM DETECTION RESULTS ON TEST SET")
    print("="*40)
    print(f"  True Positives  (Fever caught)   : {TP:,}")
    print(f"  True Negatives  (Normal normal)  : {TN:,}")
    print(f"  False Positives (False alarm)    : {FP:,}")
    print(f"  False Negatives (Missed fever)   : {FN:,}")
    print("-" * 40)
    print(f"  Precision : {precision:.3f}")
    print(f"  Recall    : {recall:.3f}")
    print(f"  F1-Score  : {f1:.3f}")
    print("="*40)
    
    print("\nModel saved to: d:/empty/biorhythm_dataset/biorhythm_lstm_model.h5")

if __name__ == "__main__":
    main()
