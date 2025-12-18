import streamlit as st
import csv, ast, os, sys, random
import numpy as np

# Gestion des imports IA
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'ia'))

try:
    from ia_predictive import analyse_combat
except ImportError:
    st.error("Fichier ia_predictive.py introuvable dans le dossier /ia")
    st.stop()

# --- INITIALISATION ---
if "deck1" not in st.session_state: st.session_state.deck1 = []
if "deck2" not in st.session_state: st.session_state.deck2 = []

def toggle_selection(deck_key, card_name):
    current = st.session_state[deck_key]
    if card_name in current:
        current.remove(card_name)
    elif len(current) < 8:
        current.append(card_name)

# --- CHARGEMENT DES DONNÉES (Chemins sécurisés) ---
@st.cache_data
def load_data():
    p_img = os.path.join(BASE_DIR, "dataset", "clashroyale_cards.csv")
    p_names = os.path.join(BASE_DIR, "dataset", "cartes.csv")
    
    img_map = {}
    if os.path.exists(p_img):
        with open(p_img, 'r', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                try:
                    urls = ast.literal_eval(r.get('iconUrls', "{}"))
                    if 'medium' in urls: img_map[r['name']] = urls['medium']
                except: pass
    
    cards = []
    if os.path.exists(p_names):
        with open(p_names, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for r in reader:
                if r and len(r) > 1:
                    cards.append({"name": r[1], "image_url": img_map.get(r[1])})
    return cards, img_map

cards_data, name_to_url = load_data()

# --- FENÊTRE DE SÉLECTION ---
# --- FENÊTRE DE SÉLECTION ---
@st.dialog("Configuration du Deck")
def open_deck_selector(key):
    c_info, c_rand, c_clear, c_close = st.columns([2, 2, 2, 1])
    with c_info: 
        st.write(f"**Cartes : {len(st.session_state[key])}/8**")
    
    with c_rand:
        if st.button("🎲 Aléatoire", use_container_width=True):
            if len(cards_data) >= 8:
                st.session_state[key] = random.sample([c['name'] for c in cards_data], 8)
                st.rerun()
            
    with c_clear: # Correction ici : plus de ":="
        if st.button("🗑️ Vider", use_container_width=True):
            st.session_state[key] = []
            st.rerun()

    with c_close:
        if st.button("✖️", type="primary", use_container_width=True): 
            st.rerun()

    search = st.text_input("🔍 Rechercher une carte...", key=f"search_{key}", label_visibility="collapsed")
    
    container = st.container(height=450)
    with container:
        cols = st.columns(4)
        filtered = [c for c in cards_data if search.lower() in c['name'].lower()]
        for i, c in enumerate(filtered):
            with cols[i % 4]:
                if c['image_url']: 
                    st.image(c['image_url'], width=70)
                
                is_sel = c['name'] in st.session_state[key]
                st.button(
                    "✅" if is_sel else "Ajouter", 
                    key=f"btn_{key}_{c['name']}",
                    type="primary" if is_sel else "secondary",
                    disabled=(len(st.session_state[key]) >= 8 and not is_sel),
                    on_click=toggle_selection, 
                    args=(key, c['name'])
                )

# --- INTERFACE DE PRÉDICTION ---
st.title("⚔️ Prédicteur de Combat")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("🔵 Deck 1 (Allié)")
    if st.button("Modifier Deck 1", key="mod1", icon="🎴"): open_deck_selector("deck1")
    d_cols = st.columns(4)
    for i, name in enumerate(st.session_state.deck1):
        with d_cols[i%4]: st.image(name_to_url.get(name, ""), caption=name, use_container_width=True)

with col2:
    st.subheader("🔴 Deck 2 (Adversaire)")
    if st.button("Modifier Deck 2", key="mod2", icon="🎴"): open_deck_selector("deck2")
    d_cols = st.columns(4)
    for i, name in enumerate(st.session_state.deck2):
        with d_cols[i%4]: st.image(name_to_url.get(name, ""), caption=name, use_container_width=True)

st.divider()

# --- SECTION ANALYSE (Autour de la ligne 125) ---
if st.button("🔮 ANALYSER LE COMBAT", type="primary", use_container_width=True):
    d1 = st.session_state.deck1
    d2 = st.session_state.deck2
    
    if len(d1) == 8 and len(d2) == 8:
        # APPEL SIMPLE AVEC SEULEMENT DEUX ARGUMENTS
        mode, p1, p2 = analyse_combat(d1, d2) 
        
        st.balloons()
        c1, c2, c3 = st.columns(3)
        c1.metric("Moteur", mode, border=True)
        c2.metric("Deck 1", f"{p1}%", border=True)
        c3.metric("Deck 2", f"{p2}%", border=True)
        st.progress(p1 / 100.0)
    else:
        st.warning("Complétez les decks (8 cartes chacun).")