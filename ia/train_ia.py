import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train():
    # --- CONFIGURATION DES CHEMINS ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset")
    
    path_combats = os.path.join(DATASET_DIR, "combats_joueurs.csv")
    path_cartes = os.path.join(DATASET_DIR, "cartes.csv")
    
    if not os.path.exists(path_combats) or not os.path.exists(path_cartes):
        print(f"❌ Erreur : Fichiers CSV introuvables dans {DATASET_DIR}")
        return

    # --- CHARGEMENT ---
    print("🚀 Chargement des données...")
    # On limite à 100k pour un bon ratio précision/vitesse (ajustable)
    df = pd.read_csv(path_combats, nrows=1000000) 
    cards_df = pd.read_csv(path_cartes)
    all_cards = cards_df.iloc[:, 1].tolist()
    
    # --- OPTIMISATION ---
    # Dictionnaire pour éviter de chercher dans une liste (très lent)
    card_to_idx = {name: i for i, name in enumerate(all_cards)}
    num_cards = len(all_cards)
    
    X = []
    y = []

    # Extraction des colonnes en matrices Numpy (beaucoup plus rapide que .iloc)
    win_decks = df[['cg1','cg2','cg3','cg4','cg5','cg6','cg7','cg8']].values
    lose_decks = df[['cp1','cp2','cp3','cp4','cp5','cp6','cp7','cp8']].values

    print(f"🧠 Préparation de l'IA ({len(df)} matchs)...")
    
    for i in range(len(df)):
        # Création de vecteurs binaires (Deck1 + Deck2)
        # On utilise int8 pour économiser 8x plus de RAM que le float64 par défaut
        vec_win_loss = np.zeros(num_cards * 2, dtype=np.int8)
        vec_loss_win = np.zeros(num_cards * 2, dtype=np.int8)
        
        for j in range(8):
            cw = win_decks[i, j]
            cl = lose_decks[i, j]
            
            # Si la carte existe dans notre référentiel
            if cw in card_to_idx:
                idx_w = card_to_idx[cw]
                vec_win_loss[idx_w] = 1
                vec_loss_win[num_cards + idx_w] = 1
                
            if cl in card_to_idx:
                idx_l = card_to_idx[cl]
                vec_win_loss[num_cards + idx_l] = 1
                vec_loss_win[idx_l] = 1
        
        # On ajoute les deux versions (Deck A gagne, Deck B perd) pour que l'IA soit neutre
        X.append(vec_win_loss)
        y.append(1)
        X.append(vec_loss_win)
        y.append(0)

    # --- ENTRAÎNEMENT ---
    print("🏗️ Entraînement du modèle (Random Forest)...")
    # n_jobs=-1 utilise TOUS tes coeurs CPU en parallèle
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=15, 
        n_jobs=-1, 
        random_state=42,
        verbose=1 # Pour voir l'avancement des arbres
    )
    
    model.fit(X, y)

    # --- SAUVEGARDE ---
    print("💾 Sauvegarde du modèle...")
    joblib.dump(model, os.path.join(DATASET_DIR, "clash_model.pkl"))
    joblib.dump(all_cards, os.path.join(DATASET_DIR, "cards_list.pkl"))
    
    print(f"✅ Terminé ! Modèle prêt dans {DATASET_DIR}")

if __name__ == "__main__":
    train()