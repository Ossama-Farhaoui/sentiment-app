import streamlit as st
import joblib

model = joblib.load("sentiment_model.pkl")

st.title("📝 Analyse de sentiment des avis Amazon")

txt = st.text_area("Entre un avis client")

if st.button("Prédire"):
    if txt.strip() == "":
        st.warning("Veuillez entrer du texte.")
    else:
        pred = model.predict([txt])[0]

        try:
            proba = model.predict_proba([txt])[0]
            conf = round(max(proba)*100, 2)
            st.success(f"Résultat : **{pred}** ({conf}% confiance)")
        except:
            st.success(f"Résultat : **{pred}**")
