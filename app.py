
import streamlit as st
import pandas as pd
from src.data.preprocessing import load_data, scale_data, split_data
from src.models.ann import build_ann
from src.eval.evaluate import evaluate_model
from src.utils.explainability import explain_model

st.title("Parkinson's Detection from EEG Data")

uploaded_file = st.file_uploader("Upload your EEG CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        X, y = load_data(uploaded_file)
        X_scaled = scale_data(X)
        X_train, X_test, y_train, y_test = split_data(X_scaled, y)

        model = build_ann(X_train.shape[1])
        model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

        st.success("Model trained successfully!")

        st.subheader("Evaluation Metrics:")
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        st.write("Accuracy:", (y_pred.flatten() == y_test).mean())

        if st.checkbox("Show SHAP Explainability"):
            explain_model(model, X_test[:100])
    except Exception as e:
        st.error(f"Error: {e}")
