from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import torch
import esm
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Initialize Flask app
app = Flask(__name__)

# Load ESM model
print("Loading ESM model...")
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
esm_model.eval()

# Load trained BiLSTM models
print("Loading trained BiLSTM models...")
model1 = load_model("bilstm_model1.h5")
model2 = load_model("bilstm_model2.h5")

# Function to extract features from ESM model
def extract_esm_features(sequences):
    batch_labels = [(str(i), seq) for i, seq in enumerate(sequences)]
    batch_tokens = batch_converter(batch_labels)[2]

    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[6])
        token_representations = results["representations"][6]

    sequence_embeddings = token_representations.mean(dim=1).cpu().numpy()
    return sequence_embeddings

@app.route("/predict", methods=["POST"])
def predict_peptide():
    try:
        data = request.get_json()
        peptide_sequence = data.get("sequence", "")
        if not peptide_sequence:
            return jsonify({"error": "Missing sequence"}), 400

        # Extract features and reshape for BiLSTM
        X_new = extract_esm_features([peptide_sequence])
        X_new = X_new[:, np.newaxis, :]

        # Make predictions
        y_pred_prob1 = model1.predict(X_new)[0][0]
        y_pred_prob2 = model2.predict(X_new)[0][0]
        y_pred_prob_avg = (y_pred_prob1 + y_pred_prob2) / 2

        # Assign label
        label = "Anti-Dengue" if y_pred_prob_avg > 0.5 else "Non-Anti-Dengue"

        return jsonify({"predicted_probability": float(y_pred_prob_avg), "prediction_label": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
