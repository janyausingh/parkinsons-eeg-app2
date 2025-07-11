
import shap
import matplotlib.pyplot as plt
import streamlit as st

def explain_model(model, X_sample):
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.summary_plot(shap_values, X_sample, show=False)
    st.pyplot(bbox_inches='tight')
