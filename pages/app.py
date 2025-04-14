import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("🩺 Health Recommender System")

# Load necessary data
with open("pages/data/mlp_model.pkl", "rb") as f:
    mlp_params = pickle.load(f)
with open("pages/data/diagnosis_encoder.pkl", "rb") as f:
    le = pickle.load(f)
with open("pages/data/drug_encoder.pkl", "rb") as f:
    mlb = pickle.load(f)

# Load CF data
cf_matrix = pd.read_csv("pages/data/cf_diagnosis_drug_matrix.csv", index_col=0)
similarity_matrix = np.load("pages/data/cf_similarity_matrix.npy")

# ---- Define CF prediction function ----
def recommend_drugs_cf(user_id, cf_matrix, similarity_matrix, k=5):
    if user_id not in cf_matrix.index:
        return []
    
    # Compute user similarity scores
    user_idx = cf_matrix.index.get_loc(user_id)
    sim_scores = similarity_matrix[user_idx]
    
    # Get top-k similar users
    top_k_users = np.argsort(sim_scores)[::-1][1:k+1]
    
    # Aggregate their drug usage
    top_users = cf_matrix.iloc[top_k_users]
    mean_scores = top_users.mean(axis=0)
    
    # Recommend drugs not already used
    user_drugs = cf_matrix.loc[user_id]
    recommendations = mean_scores[user_drugs == 0].sort_values(ascending=False).head(10)
    
    return list(recommendations.index)

# ---- Streamlit Tabs ----
tab1, tab2 = st.tabs(["🧠 Predict Diagnosis (MLP)", "🤝 Recommend Drugs (CF)"])

# --- MLP Tab ---
with tab1:
    selected_drugs = st.multiselect("Select prescribed drugs", sorted(mlb.classes_))
    if st.button("Predict Diagnosis"):
        if selected_drugs:
            x_input = mlb.transform([selected_drugs])
            def relu(x): return np.maximum(0, x)
            def softmax(x): e_x = np.exp(x - np.max(x, axis=1, keepdims=True)); return e_x / np.sum(e_x, axis=1, keepdims=True)
            Z1 = np.dot(x_input, mlp_params["W1"]) + mlp_params["b1"]
            A1 = relu(Z1)
            Z2 = np.dot(A1, mlp_params["W2"]) + mlp_params["b2"]
            probs = softmax(Z2)
            top_pred = le.inverse_transform([np.argmax(probs)])[0]
            st.success(f"🔍 Predicted Diagnosis: **{top_pred}**")
        else:
            st.warning("Please select at least one drug.")

with tab2:
    st.subheader("💊 Recommend Drugs Using CF (No Surprise)")

    # Load precomputed data
    @st.cache_data
    def load_cf_data():
        matrix = pd.read_csv("pages/data/cf_diagnosis_drug_matrix.csv", index_col=0)
        similarity = np.load("pages/data/cf_similarity_matrix.npy")
        return matrix, similarity

    cf_matrix, sim_matrix = load_cf_data()
    drug_list = cf_matrix.columns.tolist()

    # Diagnosis selection
    selected_diagnosis = st.selectbox("Select a diagnosis", cf_matrix.index)

    if st.button("Get Recommendations"):
        user_vector = cf_matrix.loc[selected_diagnosis].values

        # Get indices of drugs already prescribed (value = 1)
        given_drug_indices = np.where(user_vector == 1)[0]

        # Sum the similarity scores from the given drugs
        total_similarity = sim_matrix[given_drug_indices].sum(axis=0)

        # Zero out drugs already used
        total_similarity[given_drug_indices] = -1

        # Get top-N recommended drugs
        top_indices = np.argsort(total_similarity)[::-1][:10]
        recommended_drugs = [drug_list[i] for i in top_indices]

        st.markdown("### Recommended Drugs:")
        for drug in recommended_drugs:
            st.markdown(f"- {drug}")
