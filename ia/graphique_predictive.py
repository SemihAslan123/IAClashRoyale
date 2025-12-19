import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "graphique_donnee")


def generate_predictive_visuals():
    # Style Clash Royale (Sombre et Pro)
    plt.style.use('dark_background')

    # 1. Feature Importance (Top 10 Cartes Meta)
    imp_file = os.path.join(DATA_DIR, "feature_importance.csv")
    if os.path.exists(imp_file):
        df = pd.read_csv(imp_file).head(10)
        plt.figure(figsize=(12, 7))
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, 10))
        plt.barh(df['card'], df['importance'], color=colors, edgecolor='white')
        plt.xlabel('Poids décisionnel (Influence sur la victoire)')
        plt.title('Top 10 des cartes les plus influentes - Dataset 1.5M', fontsize=14, pad=20)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, "importance_cards.png"))
        print("✅ Graphique d'importance généré.")
        plt.show()

    # 2. Matrice de Confusion (Précision Scientifique)
    cm_file = os.path.join(DATA_DIR, "confusion_matrix.csv")
    if os.path.exists(cm_file):
        cm = pd.read_csv(cm_file).values
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                    xticklabels=['Défaite Prédite', 'Victoire Prédite'],
                    yticklabels=['Défaite Réelle', 'Victoire Réelle'],
                    cbar=False)
        plt.title('Matrice de Confusion : Performance Globale du Modèle', fontsize=14, pad=20)
        plt.ylabel('Réalité')
        plt.xlabel('Prédiction IA')
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, "confusion_matrix.png"))
        print("✅ Matrice de confusion générée.")
        plt.show()


if __name__ == "__main__":
    generate_predictive_visuals()