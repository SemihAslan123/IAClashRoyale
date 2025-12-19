import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.metrics import confusion_matrix

def train():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset")
    GRAPH_DATA_DIR = os.path.join(BASE_DIR, "..", "graphique_donnee")

    if not os.path.exists(GRAPH_DATA_DIR):
        os.makedirs(GRAPH_DATA_DIR)
    
    path_combats = os.path.join(DATASET_DIR, "combats_joueurs.csv")
    path_cartes = os.path.join(DATASET_DIR, "cartes.csv")
    
    if not os.path.exists(path_combats):
        print(f"❌ Erreur : Fichier {path_combats} non trouvé.")
        return

    print("🚀 Chargement des données (Mode XGBoost)...")
    df = pd.read_csv(path_combats, nrows=1000000) 
    all_cards = pd.read_csv(path_cartes).iloc[:, 1].tolist()
    
    card_to_idx = {name: i for i, name in enumerate(all_cards)}
    num_cards = len(all_cards)
    
    X, y = [], []
    win_decks = df[['cg1','cg2','cg3','cg4','cg5','cg6','cg7','cg8']].values
    lose_decks = df[['cp1','cp2','cp3','cp4','cp5','cp6','cp7','cp8']].values

    print(f"🧠 Préparation des vecteurs ({len(df)} matchs)...")
    for i in range(len(df)):
        v1 = np.zeros(num_cards * 2, dtype=np.int8)
        v2 = np.zeros(num_cards * 2, dtype=np.int8)
        
        for j in range(8):
            cw, cl = win_decks[i, j], lose_decks[i, j]
            if cw in card_to_idx:
                idx_w = card_to_idx[cw]
                v1[idx_w] = 1
                v2[num_cards + idx_w] = 1
            if cl in card_to_idx:
                idx_l = card_to_idx[cl]
                v1[num_cards + idx_l] = 1
                v2[idx_l] = 1
        
        X.append(v1)
        y.append(1)
        X.append(v2)
        y.append(0)

    X = np.array(X)
    y = np.array(y)

    print("🏗️ Entraînement XGBoost (Haute Précision)...")
    
    # Paramètres optimisés pour Clash Royale
    # n_estimators=500 : Plus d'étapes de correction
    # learning_rate=0.05 : On apprend doucement pour être précis
    # tree_method='hist' : Très rapide sur de gros datasets
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        tree_method='hist',
        device='cpu', # Change en 'cuda' si tu as une carte NVIDIA
        random_state=42,
        verbosity=1
    )

    model.fit(X, y)

    print("💾 Sauvegarde du modèle XGBoost...")
    joblib.dump(model, os.path.join(DATASET_DIR, "clash_model.pkl"))
    joblib.dump(all_cards, os.path.join(DATASET_DIR, "cards_list.pkl"))

    # --- EXPORT POUR GRAPHIQUES PREDICTIFS ---
    # 1. Feature Importance (Poids des cartes)
    importances = model.feature_importances_
    # On additionne l'importance "Allié" et "Ennemi" pour chaque carte
    combined_importance = importances[:num_cards] + importances[num_cards:]
    feat_df = pd.DataFrame({'card': all_cards, 'importance': combined_importance})
    feat_df.sort_values(by='importance', ascending=False).to_csv(os.path.join(GRAPH_DATA_DIR, "feature_importance.csv"), index=False)

    # 2. Matrice de Confusion (sur un échantillon pour la rapidité)
    y_pred = model.predict(X[:10000])
    cm = confusion_matrix(y[:10000], y_pred)
    pd.DataFrame(cm).to_csv(os.path.join(GRAPH_DATA_DIR, "confusion_matrix.csv"), index=False)
    
    print("✅ IA XGBoost prête et graphique aussi !")

if __name__ == "__main__":
    train()