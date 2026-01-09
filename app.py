import streamlit as st
import joblib
import pandas as pd

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Analyse de sentiment Amazon",
    page_icon="🧠",
    layout="wide"
)

# -------------------------
# Style (léger)
# -------------------------
st.markdown("""
<style>
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
.big-title { font-size: 44px; font-weight: 800; margin-bottom: 0.2rem; }
.subtitle { font-size: 18px; opacity: 0.8; margin-bottom: 1.2rem; }
.card { padding: 1rem; border-radius: 14px; border: 1px solid #eaeaea; background: #ffffff; }
.small { font-size: 14px; opacity: 0.8; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Model loading
# -------------------------
@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.pkl")

model = load_model()

# -------------------------
# Session state init
# -------------------------
if "texte" not in st.session_state:
    st.session_state.texte = ""

if "history" not in st.session_state:
    st.session_state.history = []

if "last_pred" not in st.session_state:
    st.session_state.last_pred = None

if "last_proba" not in st.session_state:
    st.session_state.last_proba = None  # (classes, proba) ou None

# -------------------------
# Header
# -------------------------
st.markdown('<div class="big-title">🧠 Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Démo NLP — Classification d’avis Amazon (TF-IDF + ML)</div>', unsafe_allow_html=True)

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("⚙️ Options")
    show_proba = st.checkbox("Afficher les probabilités", value=True)
    show_history = st.checkbox("Garder l’historique", value=True)

    st.markdown("---")
    st.subheader("✨ Exemples rapides")
    if st.button("😊 Positif"):
        st.session_state.texte = "I absolutely love this product, excellent quality!"
    if st.button("😠 Négatif"):
        st.session_state.texte = "Very disappointed, the product is broken and useless."
    if st.button("😐 Neutre"):
        st.session_state.texte = "The product is okay, nothing special."

# -------------------------
# Layout
# -------------------------
left, right = st.columns([1.2, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("✍️ Entrez un avis client")

    texte = st.text_area(
        "Texte de l’avis",
        value=st.session_state.texte,
        height=180,
        placeholder="Exemple : This product is amazing and worth the price."
    )

    colA, colB = st.columns([1, 1])
    with colA:
        predict_btn = st.button("🔮 Prédire le sentiment", use_container_width=True)
    with colB:
        clear_btn = st.button("🧹 Effacer (texte + résultat) ", use_container_width=True)

    # Effacer : vide texte + résultat, mais garde l'historique
    if clear_btn:
        st.session_state.texte = ""
        st.session_state.last_pred = None
        st.session_state.last_proba = None
        st.rerun()

    st.markdown('<div class="small">Astuce : utilise les exemples dans la sidebar pour tester vite.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Résultat")

    # Quand on clique sur prédire
    if predict_btn:
        if not texte.strip():
            st.warning("⚠️ Veuillez entrer un avis avant de lancer la prédiction.")
        else:
            # sauvegarde texte dans session_state
            st.session_state.texte = texte

            # prédiction
            pred = model.predict([texte])[0]
            st.session_state.last_pred = pred

            # probabilités si disponibles
            if show_proba:
                try:
                    proba = model.predict_proba([texte])[0]
                    classes = model.classes_
                    st.session_state.last_proba = (classes, proba)
                except Exception:
                    st.session_state.last_proba = None

            # historique (ON NE L’EFFACE JAMAIS avec Effacer)
            if show_history:
                st.session_state.history.insert(
                    0,
                    {"Texte": texte[:120], "Prédiction": pred}
                )

    # Affichage du résultat (même après rerun)
    if st.session_state.last_pred is None:
        st.info("Aucun résultat pour le moment. Entrez un avis puis cliquez sur **Prédire**.")
    else:
        pred = st.session_state.last_pred

        if pred == "positive":
            st.success("😊 **Avis POSITIF**")
        elif pred == "negative":
            st.error("😠 **Avis NÉGATIF**")
        else:
            st.info("😐 **Avis NEUTRE**")

        # Probabilités affichées proprement
        if show_proba:
            if st.session_state.last_proba is not None:
                classes, proba = st.session_state.last_proba
                confidence = round(float(max(proba)) * 100, 2)

                st.metric("Confiance du modèle", f"{confidence}%")
                st.progress(min(confidence / 100, 1.0))

                proba_df = pd.DataFrame({
                    "Classe": classes,
                    "Probabilité (%)": [round(float(p) * 100, 2) for p in proba]
                }).sort_values("Probabilité (%)", ascending=False)

                st.dataframe(proba_df, use_container_width=True, hide_index=True)
            else:
                st.caption("Probabilités non disponibles pour ce modèle.")

    # Historique
    if show_history and len(st.session_state.history) > 0:
        st.markdown("---")
        st.subheader("🕒 Historique")
        st.dataframe(
            pd.DataFrame(st.session_state.history[:10]),
            use_container_width=True,
            hide_index=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown(
    '<div class="small" style="text-align:center;">🎓 Projet NLP – Analyse de sentiment · Binôme B · TF-IDF + Machine Learning · Streamlit</div>',
    unsafe_allow_html=True
)
