import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "dataset", "clash_model.pkl")
CARDS_PATH = os.path.join(BASE_DIR, "..", "dataset", "cards_list.pkl")

model = None
all_cards = []

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    all_cards = joblib.load(CARDS_PATH)

def analyse_combat(deck1_names, deck2_names, precision=None):
    if model is None: return "Erreur", 0.0, 0.0
    
    num_cards = len(all_cards)
    card_to_idx = {name: i for i, name in enumerate(all_cards)}
    
    vector = np.zeros(num_cards * 2, dtype=np.int8)
    for name in deck1_names:
        if name in card_to_idx: vector[card_to_idx[name]] = 1
    for name in deck2_names:
        if name in card_to_idx: vector[num_cards + card_to_idx[name]] = 1

    # On demande la probabilité
    probs = model.predict_proba([vector])[0]
    
    # IMPORTANTE CORRECTION : On force le type 'float' pour Streamlit
    # probs[1] est la victoire du Deck 1, probs[0] celle du Deck 2
    p_win_d1 = float(round(probs[1] * 100, 2))
    p_win_d2 = float(round(probs[0] * 100, 2))
    
    return "XGBoost", p_win_d1, p_win_d2