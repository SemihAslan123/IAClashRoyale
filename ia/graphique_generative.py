import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "graphique_donnee")


def generate_generative_visuals():
    plt.style.use('dark_background')

    # 1. Waterfall Chart (Gain de Winrate par substitution)
    gain_file = os.path.join(DATA_DIR, "winrate_gain.csv")
    if os.path.exists(gain_file):
        df = pd.read_csv(gain_file)
        probs = df['prob'].tolist()

        # Calcul des variations pour l'effet waterfall
        diffs = [probs[0]] + [probs[i] - probs[i - 1] for i in range(1, len(probs))]

        plt.figure(figsize=(12, 6))
        cumulative = 0
        for i, (card, diff) in enumerate(zip(df['card'], diffs)):
            color = '#2ecc71' if diff >= 0 else '#e74c3c'
            plt.bar(card, diff, bottom=cumulative, color=color, edgecolor='white')
            # Affichage du score total au dessus de la barre
            cumulative += diff
            plt.text(i, cumulative + 1, f"{int(cumulative)}%", ha='center', color='white', fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Probabilité de Victoire (%)')
        plt.title('Évolution Stratégique : Gain de Winrate par mutation', fontsize=14, pad=20)
        plt.ylim(0, 110)
        plt.grid(axis='y', linestyle='--', alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, "waterfall_winrate.png"))
        print("✅ Graphique Waterfall généré.")
        plt.show()

    # 2. Histogramme Elixir (Avant vs Après)
    dist_file = os.path.join(DATA_DIR, "elixir_dist.csv")
    if os.path.exists(dist_file):
        df = pd.read_csv(dist_file)
        plt.figure(figsize=(10, 6))

        # Comparaison des distributions
        original = df[df['Type'] == 'Adversaire']['Cout']
        optimized = df[df['Type'] == 'Optimisé']['Cout']

        plt.hist(original, bins=np.arange(1, 10) - 0.5, alpha=0.4, label='Deck Adversaire', color='#e74c3c', rwidth=0.8)
        plt.hist(optimized, bins=np.arange(1, 10) - 0.5, alpha=0.7, label='Deck Optimisé (IA)', color='#f1c40f',
                 rwidth=0.5)

        plt.axvline(original.mean(), color='#e74c3c', linestyle='dashed', linewidth=2,
                    label=f'Moy. Adv: {original.mean():.1f}')
        plt.axvline(optimized.mean(), color='#f1c40f', linestyle='dashed', linewidth=2,
                    label=f'Moy. IA: {optimized.mean():.1f}')

        plt.xlabel('Coût en Élixir')
        plt.ylabel('Nombre de Cartes')
        plt.title('Équilibre de l\'Élixir : Lissage de la courbe de coût', fontsize=14, pad=20)
        plt.xticks(range(1, 10))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, "elixir_comparison.png"))
        print("✅ Graphique d'élixir généré.")
        plt.show()


if __name__ == "__main__":
    generate_generative_visuals()