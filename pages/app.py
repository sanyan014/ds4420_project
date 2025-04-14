import streamlit as st
import pickle
import numpy as np

st.title("🩺 Diagnosis Predictor Based on Prescribed Drugs")

# Load model + encoders
try:
    with open("mlp_model.pkl", "rb") as f:
        params = pickle.load(f)
    with open("diagnosis_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("drug_encoder.pkl", "rb") as f:
        mlb = pickle.load(f)

    st.success("✅ Model and encoders loaded.")

    # Show multiselect for drugs
    drug_options = sorted(mlb.classes_)
    selected_drugs = st.multiselect("Select prescribed drugs", drug_options)

    # Predict diagnosis
    if st.button("Predict Diagnosis"):
        if not selected_drugs:
            st.warning("Please select at least one drug.")
        else:
            # One-hot encode drugs
            drug_vector = mlb.transform([selected_drugs])

            # MLP forward pass
            def relu(x): return np.maximum(0, x)
            def softmax(x): 
                e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
                return e_x / np.sum(e_x, axis=1, keepdims=True)

            Z1 = np.dot(drug_vector, params["W1"]) + params["b1"]
            A1 = relu(Z1)
            Z2 = np.dot(A1, params["W2"]) + params["b2"]
            pred_probs = softmax(Z2)

            # Predicted diagnosis
            top_idx = np.argmax(pred_probs)
            predicted_diagnosis = le.inverse_transform([top_idx])[0]
            st.success(f"🔍 Predicted Diagnosis: **{predicted_diagnosis}**")

except Exception as e:
    st.error(f"🚨 Error loading model or predicting: {e}")
