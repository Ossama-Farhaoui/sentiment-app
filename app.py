import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Sentiment Amazon", page_icon="🧠", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.pkl")  # ⚠️ mets le bon nom du modèle

model = load_model()

st.markdown("# 🧠 Sentiment Analyzer")
st.write("Démo NLP — Classification d'avis Amazon")

texte = st.text_area("Entrez un avis")

if st.button("Prédire"):
    if not texte.strip():
        st.warning("Veuillez écrire un avis.")
    else:
        pred = model.predict([texte])[0]
        st.success(f"Sentiment prédit : {pred}")
