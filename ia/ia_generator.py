import joblib
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "dataset", "clash_model.pkl")
CARDS_PATH = os.path.join(BASE_DIR, "..", "dataset", "cards_list.pkl")
CARDS_INFO_PATH = os.path.join(BASE_DIR, "..", "dataset", "clashroyale_cards.csv")

# Chargement du modèle et de la liste des cartes
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
all_cards = joblib.load(CARDS_PATH) if os.path.exists(CARDS_PATH) else []

def generate_counter_deck(enemy_deck):
    if model is None or not all_cards:
        return None, 0.0

    num_cards = len(all_cards)
    card_to_idx = {name: i for i, name in enumerate(all_cards)}
    
    counter_deck = []
    current_win_prob = 0.0

    # Historique graphique Waterfall
    history = []

    # Algorithme glouton pour trouver les 8 meilleures cartes
    for slot in range(8):
        best_card_for_slot = None
        best_prob_for_slot = -1.0
        
        for card in all_cards:
            if card in counter_deck: continue  # Pas de doublons
            
            # Simulation du deck en cours de construction
            temp_deck = counter_deck + [card]
            
            # Préparation du vecteur pour XGBoost
            vector = np.zeros(num_cards * 2, dtype=np.int8)
            # Deck généré (Position alliée)
            for name in temp_deck:
                if name in card_to_idx: vector[card_to_idx[name]] = 1
            # Deck ennemi (Position adverse)
            for name in enemy_deck:
                if name in card_to_idx: vector[num_cards + card_to_idx[name]] = 1
            
            # Prédiction
            prob = model.predict_proba([vector])[0][1] # Probabilité de victoire position 1
            
            if prob > best_prob_for_slot:
                best_prob_for_slot = prob
                best_card_for_slot = card
        
        if best_card_for_slot:
            counter_deck.append(best_card_for_slot)
            current_win_prob = best_prob_for_slot
            history.append({"slot": slot + 1, "card": best_card_for_slot, "prob": round(current_win_prob * 100, 2)})

    GRAPH_DATA_DIR = os.path.join(BASE_DIR, "..", "graphique_donnee")
    if not os.path.exists(GRAPH_DATA_DIR): os.makedirs(GRAPH_DATA_DIR)

    # 1. Données Waterfall (Gain de Winrate)
    pd.DataFrame(history).to_csv(os.path.join(GRAPH_DATA_DIR, "winrate_gain.csv"), index=False)

    # 2. Données Distribution Elixir (Avant vs Après)
    if os.path.exists(CARDS_INFO_PATH):
        cards_df = pd.read_csv(CARDS_INFO_PATH)
        cost_map = dict(zip(cards_df['name'], cards_df['elixirCost']))

        original_costs = [cost_map.get(c, 0) for c in enemy_deck]
        optimized_costs = [cost_map.get(c, 0) for c in counter_deck]

        dist_df = pd.DataFrame({
            'Type': ['Adversaire'] * 8 + ['Optimisé'] * 8,
            'Cout': original_costs + optimized_costs
        })
        dist_df.to_csv(os.path.join(GRAPH_DATA_DIR, "elixir_dist.csv"), index=False)

    return counter_deck, round(current_win_prob * 100, 2)