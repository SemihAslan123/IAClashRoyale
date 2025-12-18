import streamlit as st

st.title("🪄 Générateur de Deck IA")
st.info("Cette section utilisera bientôt un modèle génératif pour créer des decks.")

st.subheader("Options de génération")
col1, col2 = st.columns(2)

with col1:
    style = st.selectbox("Style de jeu préféré", ["Beatdown", "Control", "Cycle", "Siege"])
    win_condition = st.selectbox("Condition de victoire souhaitée", ["Géant", "Chevaucheur de cochon", "Golem", "Cimetière", "Arc-X"])

with col2:
    elixir = st.slider("Coût moyen max", 2.5, 5.0, 3.8)
    priorite = st.radio("Priorité", ["Attaque", "Défense", "Équilibre"])

if st.button("🪄 Générer le Deck optimal", type="primary", use_container_width=True):
    st.warning("Le modèle d'IA générative n'est pas encore connecté.")