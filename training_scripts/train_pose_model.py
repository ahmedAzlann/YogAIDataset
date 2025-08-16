#Loads all CSV pose files.
#Encodes pose labels numerically.
#Trains a neural network to classify poses.
#Saves .h5 model & label_encoder.pkl for later use.

import pandas as pd 
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import glob
import os
import pickle

# Load all CSVs
files = glob.glob("dataset/regular/*.csv")  # change folder as needed
df = pd.concat([pd.read_csv(f) for f in files])

X = df.drop("label", axis=1)
y = df["label"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20)

# Save model & encoder
os.makedirs("models", exist_ok=True)
model.save("models/regular_model.h5")
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
