import streamlit as st
import csv
import ast
import os
import random
from ia_generator import generate_counter_deck, all_cards

# --- CONFIGURATION ---
st.set_page_config(page_title="Clash Royale AI - Counter Maker", page_icon="⚔️", layout="wide")

# --- INITIALISATION ---
if "enemy_deck" not in st.session_state: st.session_state.enemy_deck = []

# --- CHARGEMENT DES IMAGES (Récupération de votre code) ---
@st.cache_data
def load_card_images():
    path_images = "../dataset/clashroyale_cards.csv"
    image_map = {}
    if os.path.exists(path_images):
        with open(path_images, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    urls = ast.literal_eval(row.get('iconUrls', "{}"))
                    if 'medium' in urls: image_map[row.get('name')] = urls['medium']
                except: continue
    return image_map

image_map = load_card_images()

# --- MODALE DE SÉLECTION ---
@st.dialog("Sélectionner le Deck Adverse")
def open_enemy_selector():
    col_info, col_random, col_clear = st.columns([2, 2, 2])
    with col_info: st.write(f"**Cartes : {len(st.session_state.enemy_deck)}/8**")
    with col_random:
        if st.button("🎲 Aléatoire"):
            st.session_state.enemy_deck = random.sample(all_cards, 8)
            st.rerun()
    with col_clear:
        if st.button("🗑️ Vider"):
            st.session_state.enemy_deck = []
            st.rerun()

    st.divider()
    search = st.text_input("🔍 Rechercher une carte...")
    container = st.container(height=400)
    with container:
        cols = st.columns(4)
        filtered = [c for c in all_cards if search.lower() in c.lower()]
        for i, name in enumerate(filtered):
            with cols[i % 4]:
                if name in image_map: st.image(image_map[name], width=70)
                is_selected = name in st.session_state.enemy_deck
                if st.button(name, key=f"sel_{name}", type="primary" if is_selected else "secondary"):
                    if is_selected: st.session_state.enemy_deck.remove(name)
                    elif len(st.session_state.enemy_deck) < 8: st.session_state.enemy_deck.append(name)
                    st.rerun()

# --- INTERFACE ---
st.title("🛡️ Clash Royale - Générateur de Contre")

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.subheader("🔴 Deck Adverse à contrer")
    if st.button("Modifier Deck Adverse", use_container_width=True): open_enemy_selector()
    
    if st.session_state.enemy_deck:
        d_cols = st.columns(4)
        for i, name in enumerate(st.session_state.enemy_deck):
            with d_cols[i % 4]: st.image(image_map.get(name, ""), caption=name, use_container_width=True)
    else:
        st.info("Sélectionnez le deck de votre adversaire pour commencer.")

with col_right:
    st.subheader("🟢 Meilleur Contre (IA)")
    if st.session_state.enemy_deck and len(st.session_state.enemy_deck) == 8:
        if st.button("🚀 GÉNÉRER LE MEILLEUR CONTRE", type="primary", use_container_width=True):
            with st.spinner("L'IA analyse des milliers de combinaisons..."):
                counter_deck, win_prob = generate_counter_deck(st.session_state.enemy_deck)
                st.session_state.generated_deck = counter_deck
                st.session_state.win_prob = win_prob
        
        if "generated_deck" in st.session_state:
            st.metric("Probabilité de victoire estimée", f"{st.session_state.win_prob}%")
            g_cols = st.columns(4)
            for i, name in enumerate(st.session_state.generated_deck):
                with g_cols[i % 4]: st.image(image_map.get(name, ""), caption=name, use_container_width=True)
    else:
        st.write("En attente d'un deck adverse complet...")

st.divider()
st.caption("Note : La génération utilise une recherche gloutonne basée sur votre modèle XGBoost.")