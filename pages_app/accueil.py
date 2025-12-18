import streamlit as st

st.title("👑 Clash Royale AI Suite")
st.write("Bienvenue dans votre outil d'analyse et de génération assisté par intelligence artificielle.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.info("### ⚔️ Prédicteur de combat")
    st.write("Analysez les chances de victoire entre deux decks grâce à notre modèle XGBoost entraîné sur 1 million de matchs.")
    if st.button("Lancer le Prédicteur", use_container_width=True):
        st.switch_page("pages_app/prediction.py")

with col2:
    st.success("### 🪄 Générateur de Deck")
    st.write("Laissez l'IA générative composer pour vous le meilleur deck possible selon la méta actuelle.")
    if st.button("Lancer le Générateur", use_container_width=True):
        st.switch_page("pages_app/generation.py")