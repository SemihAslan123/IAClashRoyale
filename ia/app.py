import streamlit as st
import csv
import ast
import os
import sys
import numpy as np
import random

# Importation de la logique IA
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from ia_predictive import analyse_combat
except ImportError:
    st.error("Erreur : Le fichier 'ia_predictive.py' est introuvable.")
    st.stop()

# --- CONFIGURATION ---
st.set_page_config(page_title="Clash Royale AI - Predictor", page_icon="👑", layout="wide")

# --- STYLE CSS ---
st.markdown("""
    <style>
    .stButton button { width: 100%; border-radius: 10px; }
    [data-testid="stMetricValue"] { font-size: 28px; color: #ffeb3b; }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALISATION ---
if "deck1" not in st.session_state: st.session_state.deck1 = []
if "deck2" not in st.session_state: st.session_state.deck2 = []

def toggle_selection(deck_key, card_name):
    current_list = st.session_state[deck_key]
    if card_name in current_list:
        current_list.remove(card_name)
    elif len(current_list) < 8:
        current_list.append(card_name)

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_card_data():
    path_images = "../dataset/clashroyale_cards.csv"
    path_names = "../dataset/cartes.csv"
    image_map = {}
    if os.path.exists(path_images):
        with open(path_images, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    urls = ast.literal_eval(row.get('iconUrls', "{}"))
                    if 'medium' in urls: image_map[row.get('name')] = urls['medium']
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

# --- MODALE DE SÉLECTION (Correction du nom ici) ---
@st.dialog("Configuration du Deck")
def open_deck_selector(key): # Nom changé pour correspondre à l'appel
    col_info, col_random, col_clear, col_close = st.columns([2, 2, 2, 1])
    with col_info: st.write(f"**Cartes : {len(st.session_state[key])}/8**")
    
    with col_random:
        if st.button("🎲 Aléatoire", use_container_width=True):
            # Choix de 8 cartes sans doublons
            st.session_state[key] = random.sample([c['name'] for c in cards_data], 8)
            st.rerun()
            
    with col_clear:
        if st.button("🗑️ Vider", use_container_width=True):
            st.session_state[key] = []
            st.rerun()

    with col_close:
        if st.button("✖️", type="primary", use_container_width=True): st.rerun()
    
    st.divider()
    search = st.text_input("🔍 Rechercher une carte...", key=f"s_{key}", label_visibility="collapsed")
    
    container = st.container(height=450) 
    with container:
        cols = st.columns(4)
        filtered = [c for c in cards_data if search.lower() in c['name'].lower()]
        for i, c in enumerate(filtered):
            with cols[i % 4]:
                if c['image_url']: st.image(c['image_url'], width=80)
                is_selected = c['name'] in st.session_state[key]
                disabled = len(st.session_state[key]) >= 8 and not is_selected
                st.button("✅" if is_selected else "Ajouter", key=f"b_{key}_{c['name']}", 
                          type="primary" if is_selected else "secondary",
                          use_container_width=True, disabled=disabled,
                          on_click=toggle_selection, args=(key, c['name']))

# --- INTERFACE PRINCIPALE ---
st.title("⚔️ Clash Royale - Analyseur de Deck IA")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("🔵 Deck 1 (Allié)")
    if st.button("Modifier Deck 1", icon="🎴", key="m1"): open_deck_selector("deck1")
    current_d1 = st.session_state.deck1
    if current_d1:
        d_cols = st.columns(4)
        for i, name in enumerate(current_d1):
            with d_cols[i % 4]: st.image(name_to_url.get(name, ""), caption=name, use_container_width=True)
    else: st.write("Deck vide")

with col2:
    st.subheader("🔴 Deck 2 (Adversaire)")
    if st.button("Modifier Deck 2", icon="🎴", key="m2"): open_deck_selector("deck2")
    current_d2 = st.session_state.deck2
    if current_d2:
        d_cols = st.columns(4)
        for i, name in enumerate(current_d2):
            with d_cols[i % 4]: st.image(name_to_url.get(name, ""), caption=name, use_container_width=True)
    else: st.write("Deck vide")

st.markdown("---")

_, center_col, _ = st.columns([1, 2, 1])
with center_col:
    analyze_btn = st.button("🔮 PRÉDIRE LE VAINQUEUR", type="primary", use_container_width=True)

if analyze_btn:
    d1, d2 = st.session_state.deck1, st.session_state.deck2
    if len(d1) != 8 or len(d2) != 8:
        st.warning(f"⚠️ Sélectionnez 8 cartes par deck")
    else:
        with st.spinner("L'IA calcule..."):
            mode, p1, p2 = analyse_combat(d1, d2)
        
        st.balloons()
        st.markdown("### 📊 Résultats")
        r1, r2, r3 = st.columns(3)
        r1.metric("Moteur", "XGBoost", border=True)
        r2.metric("Confiance Deck 1", f"{p1}%", border=True)
        r3.metric("Confiance Deck 2", f"{p2}%", border=True)
        st.progress(float(p1) / 100.0)
        
        if p1 > p2: st.success(f"### 🏆 Victoire prédite : **Deck 1**")
        elif p2 > p1: st.error(f"### 🏆 Victoire prédite : **Deck 2**")
        else: st.info("### 🤝 Match nul parfait.")