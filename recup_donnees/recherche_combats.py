import requests
import csv
import os
import time


def load_api_key(path="cle_api.txt"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except:
        print("❌ Impossible de charger la clé API.")
        exit()


API_KEY = load_api_key()
BASE_URL = "https://api.clashroyale.com/v1"
PLAYERS_FILE = "../dataset/recherche_joueurs.csv"
OUTPUT_FILE = "../dataset/combats_joueurs.csv"
PROGRESS_FILE = "../progression/progress_combats.txt"


session = requests.Session()
session.headers.update({
    "Accept": "application/json",
    "Authorization": f"Bearer {API_KEY}"
})


# ----------------------------------------------------------
# Charger la progression
# ----------------------------------------------------------
def load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return None
    return open(PROGRESS_FILE).read().strip() or None


def save_progress(tag):
    with open(PROGRESS_FILE, "w") as f:
        f.write(tag)


# ----------------------------------------------------------
# Charger les joueurs connus
# ----------------------------------------------------------
def load_player_tags():
    tags = []
    with open(PLAYERS_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                tags.append(row[0].strip())
    return tags


# ----------------------------------------------------------
# Récupérer combats d’un joueur
# ----------------------------------------------------------
def get_battles(player_tag):
    url = f"{BASE_URL}/players/{player_tag.replace('#', '%23')}/battlelog"
    try:
        r = session.get(url, timeout=5)
        if r.status_code != 200:
            return []
        return r.json()
    except:
        return []


# ----------------------------------------------------------
# Extraire un combat complet
# ----------------------------------------------------------
def extract_battle_data(battle):

    team = battle.get("team", [{}])[0]
    opp = battle.get("opponent", [{}])[0]

    # Vérifier structure
    if "crowns" not in team or "crowns" not in opp:
        return None

    # Déterminer gagnant / perdant
    if team["crowns"] > opp["crowns"]:
        gagnant = team
        perdant = opp
    else:
        gagnant = opp
        perdant = team

    # Vérifier présence des données essentielles
    if (
        not gagnant.get("tag") or
        not perdant.get("tag") or
        "startingTrophies" not in gagnant or
        "startingTrophies" not in perdant or
        "cards" not in gagnant or
        "cards" not in perdant or
        len(gagnant["cards"]) != 8 or
        len(perdant["cards"]) != 8
    ):
        return None  # 👈 On ignore ce combat

    # Extraire cartes
    cg = [c.get("name") for c in gagnant["cards"]]
    cp = [c.get("name") for c in perdant["cards"]]

    # Vérifier que toutes les cartes ont un nom
    if any(name is None for name in cg + cp):
        return None

    return [
        gagnant["tag"],
        gagnant["startingTrophies"],
        *cg,
        perdant["tag"],
        perdant["startingTrophies"],
        *cp
    ]


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():

    print("📌 Extraction des combats complets...")

    players = load_player_tags()
    print(f"➡️ {len(players)} joueurs à traiter\n")

    last = load_progress()
    print(f"➡️ Reprise à partir de : {last}\n")

    start = (last is None)

    # Création CSV
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "jgagnant", "tropheesg",
                "cg1","cg2","cg3","cg4","cg5","cg6","cg7","cg8",
                "jperdant", "tropheesp",
                "cp1","cp2","cp3","cp4","cp5","cp6","cp7","cp8"
            ])

    output = open(OUTPUT_FILE, "a", newline="", encoding="utf-8")
    writer = csv.writer(output)

    for player_tag in players:

        if not start:
            if player_tag == last:
                start = True
            continue

        save_progress(player_tag)

        battles = get_battles(player_tag)
        print(f"✔ Joueur traité : {player_tag} ({len(battles)} combats reçus)")

        count_valid = 0

        for battle in battles:
            data = extract_battle_data(battle)
            if data:
                writer.writerow(data)
                count_valid += 1

        print(f"   → {count_valid} combats complets ajoutés")

        time.sleep(0.3)

    output.close()

    print("\n🎉 Extraction finie ! Combats complets dans combats_joueurs.csv")


if __name__ == "__main__":
    main()
