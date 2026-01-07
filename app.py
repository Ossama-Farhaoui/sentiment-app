import streamlit as st
import joblib

model = joblib.load("sentiment_model.pkl")

st.title("📝 Analyse de sentiment des avis Amazon")

texte = st.text_area("Entrer un avis client ici")

if st.button("Prédire le sentiment"):
    pred = model.predict([texte])[0]

    if pred == "positive":
        st.success("😊 Sentiment : POSITIF")
    elif pred == "negative":
        st.error("😠 Sentiment : NÉGATIF")
    else:
        st.info("😐 Sentiment : NEUTRE")

    try:
        proba = model.predict_proba([texte])[0]
        conf = round(max(proba)*100, 2)
        st.caption(f"Confiance du modèle : {conf}%")
    except:
        pass
