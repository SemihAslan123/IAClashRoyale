import joblib
import numpy as np
import os

# Chargement du modèle au démarrage
MODEL_PATH = "../dataset/clash_model.pkl"
CARDS_PATH = "../dataset/cards_list.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    all_cards = joblib.load(CARDS_PATH)
else:
    model = None
    all_cards = []

def analyse_combat(deck1_names, deck2_names, precision=None):
    """
    Utilise l'IA pour prédire l'issue du combat.
    Note: 'precision' n'est plus utilisé par l'IA car elle analyse tout le deck,
    mais on le garde pour ne pas casser ton interface Streamlit.
    """
    if model is None:
        return 0, 0, 0

    # Création du vecteur d'entrée
    vector = np.zeros(len(all_cards) * 2)
    for i, card in enumerate(all_cards):
        if card in deck1_names: vector[i] = 1
        if card in deck2_names: vector[len(all_cards) + i] = 1

    # Prédiction des probabilités
    # [probabilité de 0 (défaite D1), probabilité de 1 (victoire D1)]
    probs = model.predict_proba([vector])[0]
    
    p_win_d1 = round(probs[1] * 100, 2)
    p_win_d2 = round(probs[0] * 100, 2)
    
    # On retourne un nombre fictif de "combats" pour l'interface (100% de confiance modèle)
    return "IA Mode", p_win_d1, p_win_d2