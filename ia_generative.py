import pickle
import numpy as np
import random
import os
import streamlit as st

# Configuration des chemins absolus
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "dataset", "clash_model.pkl")
CARDS_PATH = os.path.join(BASE_DIR, "dataset", "cards_list.pkl")

# Chargement sécurisé
model = None
all_cards = []

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(CARDS_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(CARDS_PATH, 'rb') as f:
            all_cards = pickle.load(f)
    else:
        st.error(f"Fichiers manquants dans : {MODEL_PATH}")
except Exception as e:
    st.error(f"Erreur critique lors du chargement du modèle : {e}")
    st.info("Conseil d'expert : Relancez 'python3 ia/train_ia.py' pour corriger les fichiers .pkl")


def preparer_input(deck):
    """Vecteur binaire pour le RandomForest (284 colonnes)"""
    if not all_cards: return None
    num_cards = len(all_cards)
    card_to_idx = {name: i for i, name in enumerate(all_cards)}
    vec = np.zeros(num_cards * 2, dtype=np.int8)
    for card in deck:
        if card in card_to_idx:
            vec[card_to_idx[card]] = 1
    return vec.reshape(1, -1)


def generer_meilleur_deck(deck_user, iterations=50):
    """Recherche heuristique du meilleur deck via le modèle pkl"""
    if model is None: return deck_user, 0.0

    meilleur_deck = list(deck_user)
    try:
        meilleur_score = model.predict_proba(preparer_input(deck_user))[0][1]
    except:
        meilleur_score = 0

    for _ in range(iterations):
        test_deck = list(deck_user)
        n_changes = random.randint(1, 2)
        for _ in range(n_changes):
            idx = random.randint(0, 7)
            nouvelle_carte = random.choice(all_cards)
            if nouvelle_carte not in test_deck:
                test_deck[idx] = nouvelle_carte

        score = model.predict_proba(preparer_input(test_deck))[0][1]
        if score > meilleur_score:
            meilleur_score = score
            meilleur_deck = test_deck

    return meilleur_deck, meilleur_score