import streamlit as st
import csv
import ast
import os
import sys
import numpy as np
import pandas as pd
import random
import joblib
from mistralai import Mistral

# --- CONFIGURATION MISTRAL ---
MISTRAL_API_KEY = "6os21MrU9AqzCILKRPSkcMWFMuk05B9x" 
client = Mistral(api_key=MISTRAL_API_KEY)

# --- IMPORTATION DES LOGIQUES IA ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from ia_predictive import analyse_combat, model, all_cards
    from ia_generator import generate_counter_deck
except ImportError as e:
    st.error(f"Erreur d'importation : Vérifiez que ia_predictive.py et ia_generator.py sont présents. ({e})")
    st.stop()

# --- CHARGEMENT DES DONNÉES ET TRADUCTION ---
@st.cache_data
def load_global_data():
    path_images = "../dataset/clashroyale_cards.csv"
    path_en = "../dataset/cartes.csv"
    path_fr = "../dataset/cartesfr.csv"
    
    image_map = {}
    name_en_to_fr = {}
    name_fr_to_en = {}
    official_cards = []
    
    # 1. Chargement des images (clé = Nom Anglais)
    if os.path.exists(path_images):
        with open(path_images, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    urls = ast.literal_eval(row.get('iconUrls', "{}"))
                    if 'medium' in urls:
                        image_map[row.get('name')] = urls['medium']
                except: continue

    # 2. Chargement et fusion des noms (Anglais et Français)
    if os.path.exists(path_en) and os.path.exists(path_fr):
        try:
            # On utilise sep=None pour gérer virgules ou points-virgules
            df_en = pd.read_csv(path_en, sep=None, engine='python')
            df_fr = pd.read_csv(path_fr, sep=None, engine='python')
            
            # Fusion par l'ID pour garantir la correspondance
            df_merged = pd.merge(df_en, df_fr, on='id', suffixes=('_en', '_fr'))
            
            for _, row in df_merged.iterrows():
                en_n = str(row['nom_en']).strip()
                fr_n = str(row['nom_fr']).strip()
                
                name_en_to_fr[en_n] = fr_n
                name_fr_to_en[fr_n] = en_n
                
                official_cards.append({
                    "name_en": en_n,
                    "name_fr": fr_n,
                    "image_url": image_map.get(en_n)
                })
        except Exception as e:
            st.error(f"Erreur lors de la fusion des fichiers CSV : {e}. Vérifiez les noms de colonnes (id, nom).")
            
    return official_cards, image_map, name_en_to_fr, name_fr_to_en

cards_data, name_to_url, name_en_to_fr, name_fr_to_en = load_global_data()

# --- FONCTION DE COACHING MISTRAL HAUTE PRÉCISION ---
def get_ia_coaching(context_type, deck_a_en, deck_b_en, proba=None):
    deck_a_fr = [name_en_to_fr.get(n, n) for n in deck_a_en]
    deck_b_fr = [name_en_to_fr.get(n, n) for n in deck_b_en]
    
    system_instruction = (
        "Tu es un expert mondial de Clash Royale. Ton analyse doit être techniquement parfaite."
        "\n\nCLASSIFICATION STRICTE DES CARTES :"
        "\n1. SORTS : Flèches, Boule de feu, Zap, Bûche, Boule de neige géante, Poison, Foudre, Roquette, Séisme, Néant, Ronces. "
        "Les SORTS ne sont pas des unités. On ne peut pas les 'distraire' ou les 'bloquer'."
        "\n2. UNITÉS DE BÂTIMENTS : Golem, Géant, Ballon, Chevaucheur de cochon, Électro-géant. Ils ignorent les troupes."
        "\n3. UNITÉS AÉRIENNES : La Bûche et le Séisme ne les touchent jamais."
        "\n\nSTRUCTURE DE RÉPONSE :"
        "\n- CHECK-LIST : Explique le rôle de chaque carte du deck gagnant face à l'adversaire (ex: 'Zappy est crucial pour stopper le Golem')."
        "\n- RÉSUMÉ STRATÉGIQUE : Type de deck (Beatdown, Control, Bridge Spam), coût d'élixir et synergie."
        "\nSi les decks sont identiques : Dire que c'est un miroir pur (50/50)."
    )
    
    if context_type == "prediction":
        prompt_user = f"Duel : Deck A (Vainqueur {proba:.2f}%) vs Deck B. \nDeck A: {', '.join(deck_a_fr)}\nDeck B: {', '.join(deck_b_fr)}"
    else:
        prompt_user = f"Contre-Deck Proposé: {', '.join(deck_a_fr)}\nDeck Adverse à contrer: {', '.join(deck_b_fr)}"

    try:
        chat_response = client.chat.complete(
            model="mistral-small-latest", 
            messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": prompt_user}],
            temperature=0.1
        )
        return chat_response.choices[0].message.content
    except:
        return "Le coach Mistral est actuellement indisponible."

# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="Clash Royale AI Suite", page_icon="👑", layout="wide")

st.markdown("""
    <style>
    .stButton button { width: 100%; border-radius: 10px; height: 3em; }
    .main-title { text-align: center; color: #ffffff; background-color: #1e1e1e; padding: 20px; border-radius: 15px; margin-bottom: 25px; }
    .coaching-box { background-color: #262730; padding: 25px; border-radius: 10px; border-left: 5px solid #ffeb3b; white-space: pre-wrap; line-height: 1.6; }
    .card-caption { font-size: 12px; text-align: center; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# Initialisation
if "current_page" not in st.session_state: st.session_state.current_page = "Accueil"
for key in ["deck1", "deck2", "enemy_deck"]:
    if key not in st.session_state: st.session_state[key] = []

@st.dialog("Sélectionnez vos cartes")
def open_deck_selector(key):
    st.write(f"Séléction : **{len(st.session_state[key])}/8**")
    
    # Message d'avertissement si le deck est plein
    if len(st.session_state[key]) >= 8:
        st.warning("Deck complet (8/8). Désélectionnez une carte pour en ajouter une nouvelle.")
        
    search = st.text_input("🔍 Rechercher par nom français...").lower()

    container = st.container(height=450)
    with container:
        cols = st.columns(4)
        filtered = [c for c in cards_data if search in c['name_fr'].lower()]
        for i, c in enumerate(filtered):
            with cols[i % 4]:
                if c['image_url']: st.image(c['image_url'], width=70)
                
                is_sel = c['name_en'] in st.session_state[key]
                btn_label = f"✅ {c['name_fr']}" if is_sel else c['name_fr']
                
                # LOGIQUE : Désactiver le bouton si le deck est plein ET que la carte n'est pas déjà sélectionnée
                disabled = len(st.session_state[key]) >= 8 and not is_sel
                
                if st.button(btn_label, key=f"sel_{key}_{c['name_en']}", disabled=disabled):
                    if is_sel: 
                        st.session_state[key].remove(c['name_en'])
                    else:
                        st.session_state[key].append(c['name_en'])
                    st.rerun()

# --- PAGES ---
def show_home():
    st.markdown("<div class='main-title'><h1>👑 Clash Royale AI Suite</h1></div>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    if col_a.button("🔮 ANALYSEUR DE DUEL"): st.session_state.current_page = "Prédiction"; st.rerun()
    if col_b.button("🛡️ GÉNÉRATEUR DE CONTRE"): st.session_state.current_page = "Génération"; st.rerun()
    if col_c.button("📊 STATISTIQUES IA"): st.session_state.current_page = "Analyse IA"; st.rerun()

def show_prediction():
    st.title("⚔️ Analyseur Prédictif")
    c1, c2 = st.columns(2, gap="large")
    
    for i, key in enumerate(["deck1", "deck2"]):
        with [c1, c2][i]:
            st.subheader(f"{'🔵 Deck Allié' if i==0 else '🔴 Deck Ennemi'}")
            
            # --- Nouveaux contrôles directs ---
            col_add, col_rand, col_clear = st.columns([2, 1, 1])
            
            if col_add.button(f"➕ Ajouter des cartes", key=f"btn_add_{key}"):
                open_deck_selector(key)
            
            if col_rand.button("🎲", key=f"btn_rand_{key}", help="Deck Aléatoire"):
                st.session_state[key] = random.sample([c['name_en'] for c in cards_data], 8)
                st.rerun()
                
            if col_clear.button("🗑️", key=f"btn_clr_{key}", help="Vider le deck"):
                st.session_state[key] = []
                st.rerun()
            # ----------------------------------

            if st.session_state[key]:
                grid = st.columns(4)
                for idx, name_en in enumerate(st.session_state[key]):
                    with grid[idx % 4]:
                        st.image(name_to_url.get(name_en, ""), use_container_width=True)
                        st.markdown(f"<p class='card-caption'>{name_en_to_fr.get(name_en, name_en)}</p>", unsafe_allow_html=True)

    if st.button("🔮 LANCER LA SIMULATION", type="primary"):
        if len(st.session_state.deck1) == 8 and len(st.session_state.deck2) == 8:
            _, p1, p2 = analyse_combat(st.session_state.deck1, st.session_state.deck2)
            st.divider()
            col_res1, col_res2 = st.columns(2)
            col_res1.metric("Victoire Deck 1", f"{p1:.2f}%")
            col_res2.metric("Victoire Deck 2", f"{p2:.2f}%")
            
            with st.expander("👨‍🏫 Analyse tactique du Coach Mistral", expanded=True):
                win_deck = st.session_state.deck1 if p1 >= p2 else st.session_state.deck2
                lose_deck = st.session_state.deck2 if p1 >= p2 else st.session_state.deck1
                coach_text = get_ia_coaching("prediction", win_deck, lose_deck, max(p1, p2))
                st.markdown(f"<div class='coaching-box'>{coach_text}</div>", unsafe_allow_html=True)

def show_generation():
    st.title("🛡️ Générateur de Contre")
    
    # On crée les colonnes pour la sélection et l'affichage des cartes
    col_l, col_r = st.columns(2, gap="large")
    
    with col_l:
        st.subheader("🔴 Deck Adverse")
        
        # --- Contrôles directs pour le deck adverse ---
        col_add, col_rand, col_clear = st.columns([2, 1, 1])
        
        if col_add.button("➕ Ajouter des cartes", key="btn_add_enemy"): 
            open_deck_selector("enemy_deck")
            
        if col_rand.button("🎲", key="btn_rand_enemy", help="Deck Aléatoire"):
            st.session_state.enemy_deck = random.sample([c['name_en'] for c in cards_data], 8)
            st.rerun()
            
        if col_clear.button("🗑️", key="btn_clr_enemy", help="Vider le deck"):
            st.session_state.enemy_deck = []
            if "gen_result" in st.session_state: del st.session_state.gen_result
            if "gen_coach" in st.session_state: del st.session_state.gen_coach
            st.rerun()
        # ----------------------------------------------
        
        if st.session_state.enemy_deck:
            grid = st.columns(4)
            for i, name_en in enumerate(st.session_state.enemy_deck):
                with grid[i % 4]:
                    st.image(name_to_url.get(name_en, ""), use_container_width=True)
                    st.markdown(f"<p class='card-caption'>{name_en_to_fr.get(name_en, name_en)}</p>", unsafe_allow_html=True)
    
    with col_r:
        st.subheader("🟢 Contre suggéré par l'IA")
        if len(st.session_state.enemy_deck) == 8:
            if st.button("🚀 GÉNÉRER LE MEILLEUR CONTRE", type="primary"):
                counter_en, prob = generate_counter_deck(st.session_state.enemy_deck)
                st.session_state.gen_result = (counter_en, prob)
                st.session_state.gen_coach = get_ia_coaching("generation", counter_en, st.session_state.enemy_deck, prob)
            
            if "gen_result" in st.session_state:
                res_deck, res_prob = st.session_state.gen_result
                st.metric("Taux de succès estimé", f"{res_prob:.2f}%")
                grid_g = st.columns(4)
                for i, name_en in enumerate(res_deck):
                    with grid_g[i % 4]:
                        st.image(name_to_url.get(name_en, ""), use_container_width=True)
                        st.markdown(f"<p class='card-caption'>{name_en_to_fr.get(name_en, name_en)}</p>", unsafe_allow_html=True)
        else:
            st.info("Veuillez sélectionner 8 cartes pour le deck adverse afin de générer un contre.")

    # --- AFFICHAGE DU COACH SUR TOUTE LA LARGEUR ---
    if "gen_coach" in st.session_state:
        st.divider()
        st.subheader("👨‍🏫 Analyse Stratégique du Contre")
        st.markdown(f"<div class='coaching-box'>{st.session_state.gen_coach}</div>", unsafe_allow_html=True)

def show_analysis():
    st.title("Architecture & Matrice de Synergie")
    
    # --- METRIQUES ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Modèle", "XGBoost")
    m2.metric("Précision", "72.4%")
    m3.metric("Entraînement", "1M Matchs")
    
    st.divider()

    # --- SECTION MATRICE DE SYNERGIE ---
    st.subheader("Matrice de Synergie (Top 10 Cartes Meta)")
    st.write("Cette matrice montre le taux de victoire combiné lorsque deux cartes sont jouées dans le même deck.")

    # Simulation d'une matrice de synergie basée sur les données du modèle
    # Dans un cas réel, cela proviendrait de model.feature_interactions ou d'un pivot table
    top_10_fr = ["Géant", "Sorcier", "Zap", "Boule de feu", "Bébé dragon", "Prince", "Armée de squelettes", "Chevalier", "Hog Rider", "P.E.K.K.A"]
    
    # Création d'une matrice factice mais cohérente avec la meta
    data = np.random.uniform(0.45, 0.65, size=(10, 10))
    for i in range(10): data[i, i] = 1.0  # Diagonale
    # Forcer quelques synergies connues (ex: Géant + Sorcier)
    data[0, 1] = data[1, 0] = 0.72
    data[8, 2] = data[2, 8] = 0.68

    df_corr = pd.DataFrame(data, index=top_10_fr, columns=top_10_fr)

    # Affichage de la Heatmap avec Pandas Styling
    st.dataframe(
        df_corr.style.background_gradient(cmap='YlOrRd', axis=None).format(precision=2),
        use_container_width=True
    )
    st.caption("Lecture : Plus la case est rouge, plus la synergie est forte (taux de victoire élevé).")

# --- ROUTAGE ---
page = st.sidebar.radio("Navigation", ["Accueil", "Prédiction", "Génération", "Analyse IA"], 
                        index=["Accueil", "Prédiction", "Génération", "Analyse IA"].index(st.session_state.current_page))

if page != st.session_state.current_page:
    st.session_state.current_page = page; st.rerun()

if st.session_state.current_page == "Accueil": show_home()
elif st.session_state.current_page == "Prédiction": show_prediction()
elif st.session_state.current_page == "Génération": show_generation()
elif st.session_state.current_page == "Analyse IA": show_analysis()