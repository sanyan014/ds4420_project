import streamlit as st

st.set_page_config(page_title="Personalized Health Recommender", layout="centered")

st.title("➕ Personalized Health Recommender System")
st.subheader("A machine learning approach to predicting diagnosis from prescribed drugs")

# Introduction
st.markdown("""
### 📌 Introduction
This project aims to develop a **personalized health recommendation system** that predicts potential diagnoses or treatments using patient-specific features.

We explore:
- **Collaborative Filtering (CF)** to recommend treatments
- **Multi-Layer Perceptron (MLP)** to classify diagnoses based on prescribed medications

The goal is to assist healthcare providers with **smart clinical suggestions** that may enhance decision-making.

---
""")

# Methodology
st.markdown("""
### 🔬 Methodology
We used a subset of the **MIMIC-III Clinical Database**, focusing on patients with both prescription and diagnosis records.

Steps included:
- Preprocessing: cleaning, encoding, transforming drugs into a multi-hot matrix
- Training: used MLP to classify diagnoses based on prescribed drugs
- Evaluation: tracked accuracy, loss, precision, recall, and F1-score

---

### ⚠️ Current Limitations
Unfortunately, our dataset was **too small and imbalanced**, so the MLP model was unable to capture reliable patterns:
- Most predictions default to a single class
- Precision and recall are low across classes
- Accuracy improves slightly during training but never stabilizes well

We believe that with **a larger, more balanced dataset**, the model could generalize much better and learn more meaningful relationships between prescriptions and diagnoses.
""")

# Future work
st.markdown("""
### 🔮 Future Work
- Incorporate more features (e.g., lab results, procedures)
- Use modern architectures like **Transformers** for better text understanding
- Train on a more complete version of the MIMIC-III dataset
""")

# Link to prediction app
st.markdown("---")
st.subheader("🔍 Try Our Demo!")
if st.button("Launch Diagnosis Predictor App"):
    st.switch_page("pages/app.py")  # make sure `app.py` is your prediction interface
