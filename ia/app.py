import streamlit as st
import csv
import ast
import os
import sys
import numpy as np

# Importation de la logique IA
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from ia_predictive import analyse_combat
except ImportError:
    st.error("Erreur : Le fichier 'ia_predictive.py' est introuvable dans le répertoire.")
    st.stop()

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Clash Royale AI - Predictor",
    page_icon="👑",
    layout="wide"
)

# --- STYLE CSS PERSONNALISÉ ---
st.markdown("""
    <style>
    .stButton button { width: 100%; border-radius: 10px; }
    .card-container { text-align: center; margin-bottom: 20px; }
    [data-testid="stMetricValue"] { font-size: 28px; color: #ffeb3b; }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALISATION ---
if "deck1" not in st.session_state:
    st.session_state.deck1 = []
if "deck2" not in st.session_state:
    st.session_state.deck2 = []

def toggle_selection(deck_key, card_name):
    current_list = st.session_state[deck_key]
    if card_name in current_list:
        current_list.remove(card_name)
    else:
        if len(current_list) < 8:
            current_list.append(card_name)
        else:
            st.toast(f"Le {deck_key} est plein !", icon="⚠️")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_card_data():
    path_images = "dataset/clashroyale_cards.csv"
    path_names = "dataset/cartes.csv"
    image_map = {}

    if os.path.exists(path_images):
        with open(path_images, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    urls = ast.literal_eval(row.get('iconUrls', "{}"))
                    if 'medium' in urls:
                        image_map[row.get('name')] = urls['medium']
                except: continue

    official_cards = []
    if os.path.exists(path_names):
        with open(path_names, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    name = row[1]
                    official_cards.append({"name": name, "image_url": image_map.get(name)})
    return official_cards

cards_data = load_card_data()
name_to_url = {c["name"]: c["image_url"] for c in cards_data}

# --- MODALE DE SÉLECTION ---
@st.dialog("Sélectionner les cartes du deck")
def open_deck_selector(deck_key):
    count = len(st.session_state[deck_key])
    st.markdown(f"**Cartes sélectionnées : {count} / 8**")
    
    search = st.text_input("🔍 Rechercher une carte...", key=f"search_{deck_key}")
    filtered_cards = [c for c in cards_data if search.lower() in c["name"].lower()]

    cols = st.columns(4)
    for i, card in enumerate(filtered_cards):
        c_name = card["name"]
        c_url = card["image_url"]
        is_selected = c_name in st.session_state[deck_key]

        with cols[i % 4]:
            if c_url: st.image(c_url, use_container_width=True)
            else: st.caption(c_name)
            
            btn_label = "✅" if is_selected else "Choisir"
            st.button(btn_label, key=f"btn_{deck_key}_{c_name}", 
                      type="primary" if is_selected else "secondary",
                      disabled=(count >= 8 and not is_selected),
                      on_click=toggle_selection, args=(deck_key, c_name))

    if st.button("Valider le deck", type="primary"):
        st.rerun()

# --- INTERFACE PRINCIPALE ---
st.title("⚔️ Clash Royale - Analyseur de Deck IA")
st.info("Cette IA prédit l'issue d'un match en analysant les synergies et les counters via un modèle Machine Learning.")

col1, col2 = st.columns(2, gap="large")

# Deck 1 (Bleu)
with col1:
    st.subheader("🔵 Deck 1 (Allié)")
    if st.button("Modifier Deck 1", icon="🎴", key="m1"):
        open_deck_selector("deck1")
    
    current_d1 = st.session_state.deck1
    if current_d1:
        d_cols = st.columns(4)
        for i, name in enumerate(current_d1):
            with d_cols[i % 4]:
                st.image(name_to_url.get(name, ""), caption=name, use_container_width=True)
    else: st.write("Deck vide")

# Deck 2 (Rouge)
with col2:
    st.subheader("🔴 Deck 2 (Adversaire)")
    if st.button("Modifier Deck 2", icon="🎴", key="m2"):
        open_deck_selector("deck2")
    
    current_d2 = st.session_state.deck2
    if current_d2:
        d_cols = st.columns(4)
        for i, name in enumerate(current_d2):
            with d_cols[i % 4]:
                st.image(name_to_url.get(name, ""), caption=name, use_container_width=True)
    else: st.write("Deck vide")

st.markdown("---")

# --- ACTION DE PRÉDICTION ---
_, center_col, _ = st.columns([1, 2, 1])
with center_col:
    analyze_btn = st.button("🔮 PRÉDIRE LE VAINQUEUR", type="primary", use_container_width=True)

if analyze_btn:
    d1 = st.session_state.deck1
    d2 = st.session_state.deck2

    if len(d1) != 8 or len(d2) != 8:
        st.warning(f"⚠️ Sélectionnez 8 cartes par deck (D1: {len(d1)} | D2: {len(d2)})")
    else:
        with st.spinner("L'IA calcule les probabilités..."):
            mode, p1, p2 = analyse_combat(d1, d2)

        if mode == 0 and p1 == 0:
            st.error("❌ Modèle IA non trouvé ! Veuillez lancer `train_ia.py` d'abord.")
        else:
            st.balloons()
            st.markdown("### 📊 Résultats de la prédiction")
            r1, r2, r3 = st.columns(3)
            r1.metric("Moteur", "Machine Learning", border=True)
            r2.metric("Confiance Deck 1", f"{p1}%", border=True)
            r3.metric("Confiance Deck 2", f"{p2}%", border=True)

            # Barre de progression visuelle
            st.progress(p1 / 100)
            
            if p1 > p2:
                st.success(f"### 🏆 Victoire prédite : **Deck 1** ({p1}%)")
            elif p2 > p1:
                st.error(f"### 🏆 Victoire prédite : **Deck 2** ({p2}%)")
            else:
                st.info("### 🤝 Match nul parfait selon l'IA.")