import streamlit as st
import csv
import ast
import os
import sys
import numpy as np
import random
from mistralai import Mistral

# --- CONFIGURATION MISTRAL ---
MISTRAL_API_KEY = "6os21MrU9AqzCILKRPSkcMWFMuk05B9x"
client = Mistral(api_key=MISTRAL_API_KEY)

# --- IMPORTATION DES LOGIQUES IA ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from ia_predictive import analyse_combat
    from ia_generator import generate_counter_deck, all_cards
except ImportError as e:
    st.error(f"Erreur d'importation : Vérifiez que 'ia_predictive.py' et 'ia_generator.py' sont dans le même dossier. ({e})")
    st.stop()

# --- FONCTION DE COACHING MISTRAL ---
def get_ia_coaching(context_type, deck_a, deck_b, proba=None):
    """Génère une explication stratégique via Mistral AI."""
    if context_type == "prediction":
        prompt = f"""
        En tant qu'expert Clash Royale, explique pourquoi le Deck A a {proba}% de chances de gagner contre le Deck B.
        Deck A : {', '.join(deck_a)}
        Deck B : {', '.join(deck_b)}
        Donne 3 points clés stratégiques (contre-attaques, synergies, win condition). Sois bref, pro et technique.
        """
    else:
        prompt = f"""
        En tant qu'expert Clash Royale, explique pourquoi ce deck de contre est efficace contre le deck adverse.
        Deck Adverse : {', '.join(deck_a)}
        Deck de Contre proposé : {', '.join(deck_b)}
        Explique les interactions spécifiques (ex: quel sort contre quel bâtiment, quelle unité stoppe leur win condition).
        """
    try:
        chat_response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        return chat_response.choices[0].message.content
    except Exception as e:
        return f"Le coach Mistral est indisponible (Erreur API). Détail : {e}"

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Clash Royale AI Suite", page_icon="👑", layout="wide")

# --- STYLE CSS ---
st.markdown("""
    <style>
    .stButton button { width: 100%; border-radius: 10px; }
    [data-testid="stMetricValue"] { font-size: 28px; color: #ffeb3b; }
    .main-title { text-align: center; color: #ffffff; background-color: #1e1e1e; padding: 20px; border-radius: 15px; margin-bottom: 25px; }
    .coaching-box { background-color: #262730; padding: 20px; border-radius: 10px; border-left: 5px solid #ffeb3b; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_global_data():
    path_images = "../dataset/clashroyale_cards.csv"
    path_names = "../dataset/cartes.csv"
    image_map = {}
    official_cards = []
    
    if os.path.exists(path_images):
        with open(path_images, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    urls = ast.literal_eval(row.get('iconUrls', "{}"))
                    if 'medium' in urls: image_map[row.get('name')] = urls['medium']
                except: continue

    if os.path.exists(path_names):
        with open(path_names, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    name = row[1]
                    official_cards.append({"name": name, "image_url": image_map.get(name)})
    return official_cards, image_map

cards_data, name_to_url = load_global_data()

# --- INITIALISATION DES ETATS ---
for key in ["deck1", "deck2", "enemy_deck"]:
    if key not in st.session_state: st.session_state[key] = []

def toggle_selection(deck_key, card_name):
    if deck_key == "enemy_deck" and "gen_result" in st.session_state:
        del st.session_state["gen_result"]
        if "gen_analysis" in st.session_state: del st.session_state["gen_analysis"]
        
    current_list = st.session_state[deck_key]
    if card_name in current_list:
        current_list.remove(card_name)
    elif len(current_list) < 8:
        current_list.append(card_name)

# --- MODALE DE SÉLECTION ---
@st.dialog("Configuration du Deck")
def open_deck_selector(key):
    col_info, col_random, col_clear, col_close = st.columns([2, 2, 2, 1])
    with col_info: st.write(f"**Cartes : {len(st.session_state[key])}/8**")
    
    with col_random:
        if st.button("🎲 Aléatoire", key=f"rnd_{key}"):
            st.session_state[key] = random.sample([c['name'] for c in cards_data], 8)
            if key == "enemy_deck":
                if "gen_result" in st.session_state: del st.session_state["gen_result"]
                if "gen_analysis" in st.session_state: del st.session_state["gen_analysis"]
            st.rerun()
            
    with col_clear:
        if st.button("🗑️ Vider", key=f"clr_{key}"):
            st.session_state[key] = []
            if key == "enemy_deck":
                if "gen_result" in st.session_state: del st.session_state["gen_result"]
                if "gen_analysis" in st.session_state: del st.session_state["gen_analysis"]
            st.rerun()
            
    with col_close:
        if st.button("✖️", type="primary", key=f"cls_{key}"): st.rerun()
    
    st.divider()
    search = st.text_input("🔍 Rechercher une carte...", key=f"search_input_{key}")
    container = st.container(height=450) 
    with container:
        cols = st.columns(4)
        filtered = [c for c in cards_data if search.lower() in c['name'].lower()]
        for i, c in enumerate(filtered):
            with cols[i % 4]:
                if c['image_url']: st.image(c['image_url'], width=80)
                is_selected = c['name'] in st.session_state[key]
                disabled = len(st.session_state[key]) >= 8 and not is_selected
                st.button("✅" if is_selected else "Ajouter", key=f"btn_{key}_{c['name']}", 
                          type="primary" if is_selected else "secondary",
                          use_container_width=True, disabled=disabled,
                          on_click=toggle_selection, args=(key, c['name']))

# --- PAGES ---
def show_home():
    st.markdown("<div class='main-title'><h1>👑 Clash Royale AI Suite</h1><p>Optimisez vos combats avec l'intelligence artificielle</p></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.info("### 🔮 Analyseur de Combat")
        if st.button("Lancer la Prédiction", key="go_p"): st.session_state.current_page = "Prédiction" ; st.rerun()
    with col2:
        st.success("### 🛡️ Générateur de Contre")
        if st.button("Lancer la Génération", key="go_g"): st.session_state.current_page = "Génération" ; st.rerun()

def show_prediction():
    st.title("⚔️ Clash Royale - Analyseur Prédictif")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("🔵 Deck 1 (Allié)")
        if st.button("Modifier Deck 1", icon="🎴", key="m1"): open_deck_selector("deck1")
        if st.session_state.deck1:
            d_cols = st.columns(4)
            for i, name in enumerate(st.session_state.deck1):
                with d_cols[i % 4]: st.image(name_to_url.get(name, ""), caption=name, use_container_width=True)
    with col2:
        st.subheader("🔴 Deck 2 (Adversaire)")
        if st.button("Modifier Deck 2", icon="🎴", key="m2"): open_deck_selector("deck2")
        if st.session_state.deck2:
            d_cols = st.columns(4)
            for i, name in enumerate(st.session_state.deck2):
                with d_cols[i % 4]: st.image(name_to_url.get(name, ""), caption=name, use_container_width=True)

    st.divider()
    if st.button("🔮 PRÉDIRE LE VAINQUEUR", type="primary"):
        if len(st.session_state.deck1) == 8 and len(st.session_state.deck2) == 8:
            mode, p1, p2 = analyse_combat(st.session_state.deck1, st.session_state.deck2)
            st.metric("Confiance Deck 1", f"{p1}%")
            st.metric("Confiance Deck 2", f"{p2}%")
            
            with st.expander("👨‍🏫 L'avis du Coach Mistral", expanded=True):
                with st.spinner("Analyse stratégique en cours..."):
                    winner = st.session_state.deck1 if p1 > p2 else st.session_state.deck2
                    loser = st.session_state.deck2 if p1 > p2 else st.session_state.deck1
                    analysis = get_ia_coaching("prediction", winner, loser, max(p1, p2))
                    st.markdown(f"<div class='coaching-box'>{analysis}</div>", unsafe_allow_html=True)
        else:
            st.warning("Veuillez sélectionner 8 cartes pour chaque deck.")

def show_generation():
    st.title("🛡️ Clash Royale - Générateur de Contre")
    col_left, col_right = st.columns(2, gap="large")
    
    with col_left:
        st.subheader("🔴 Deck Adverse à contrer")
        if st.button("Modifier Deck Adverse", use_container_width=True, key="m_adv"): open_deck_selector("enemy_deck")
        if st.session_state.enemy_deck:
            d_cols = st.columns(4)
            for i, name in enumerate(st.session_state.enemy_deck):
                with d_cols[i % 4]: st.image(name_to_url.get(name, ""), caption=name, use_container_width=True)

    with col_right:
        st.subheader("🟢 Meilleur Contre (IA)")
        if len(st.session_state.enemy_deck) == 8:
            if st.button("🚀 GÉNÉRER LE MEILLEUR CONTRE", type="primary", use_container_width=True):
                with st.spinner("L'IA calcule le deck optimal..."):
                    counter_deck, win_prob = generate_counter_deck(st.session_state.enemy_deck)
                    st.session_state.gen_result = (counter_deck, win_prob)
                    # Génération automatique du coaching après la création du deck
                    st.session_state.gen_analysis = get_ia_coaching("generation", st.session_state.enemy_deck, counter_deck)
            
            if "gen_result" in st.session_state:
                res_deck, res_prob = st.session_state.gen_result
                st.metric("Probabilité de victoire", f"{res_prob}%")
                g_cols = st.columns(4)
                for i, name in enumerate(res_deck):
                    with g_cols[i % 4]: st.image(name_to_url.get(name, ""), caption=name, use_container_width=True)
                
                if "gen_analysis" in st.session_state:
                    st.markdown(f"<div class='coaching-box'><strong>👨‍🏫 Analyse du Coach :</strong><br><br>{st.session_state.gen_analysis}</div>", unsafe_allow_html=True)
            else:
                st.info("Le deck adverse a été modifié. Veuillez générer un nouveau contre.")
        else:
            st.write("En attente d'un deck adverse complet...")

# --- NAVIGATION ---
if "current_page" not in st.session_state: st.session_state.current_page = "Accueil"

st.sidebar.title("🎮 Navigation")
page_choice = st.sidebar.radio("Aller vers :", ["Accueil", "Prédiction", "Génération"], 
                               index=["Accueil", "Prédiction", "Génération"].index(st.session_state.current_page))

if page_choice != st.session_state.current_page:
    st.session_state.current_page = page_choice
    st.rerun()

if st.session_state.current_page == "Accueil": show_home()
elif st.session_state.current_page == "Prédiction": show_prediction()
elif st.session_state.current_page == "Génération": show_generation()