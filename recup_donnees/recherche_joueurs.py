import requests
import csv
import os


def load_api_key(path="cle_api.txt"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print("❌ Fichier cle_api.txt introuvable.")
        exit()


API_KEY = load_api_key()
BASE_URL = "https://api.clashroyale.com/v1"

CLANS_FILE = "../dataset/clans_trouves.csv"
PLAYERS_FILE = "../dataset/recherche_joueurs.csv"
PROGRESS_FILE = "../progression/progress_joueurs.txt"

session = requests.Session()
session.headers.update({
    "Accept": "application/json",
    "Authorization": f"Bearer {API_KEY}"
})


# -------------------------------------------------------------
#  Charger les clans validés
# -------------------------------------------------------------
def load_valid_clans(path):
    clans = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2 and row[1].strip().lower() == "valide":
                clans.append(row[0].strip())
    return clans


# -------------------------------------------------------------
#  Gestion de la progression
# -------------------------------------------------------------
def load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return None
    return open(PROGRESS_FILE, "r").read().strip() or None


def save_progress(tag):
    with open(PROGRESS_FILE, "w") as f:
        f.write(tag)


# -------------------------------------------------------------
#  API : récupérer les joueurs d’un clan
# -------------------------------------------------------------
def get_clan_members(tag):
    try:
        url = f"{BASE_URL}/clans/{tag.replace('#', '%23')}"
        r = session.get(url, timeout=4)

        if r.status_code != 200:
            return []

        data = r.json()
        return data.get("memberList", [])

    except:
        return []


# -------------------------------------------------------------
#  Main
# -------------------------------------------------------------
def main():
    print("📌 Extraction des joueurs en cours...")

    # Charger les clans valides
    clans = load_valid_clans(CLANS_FILE)
    print(f"➡️ {len(clans)} clans valides trouvés.\n")

    # Charger progression
    last = load_progress()
    print(f"➡️ Reprise à partir de : {last}\n")

    start = (last is None)

    # Charger joueurs déjà enregistrés
    known_players = set()
    if os.path.exists(PLAYERS_FILE):
        with open(PLAYERS_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                known_players.add(row[0].strip())
    else:
        with open(PLAYERS_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["PlayerTag"])

    player_file = open(PLAYERS_FILE, "a", newline="", encoding="utf-8")
    writer = csv.writer(player_file)

    for clan_tag in clans:

        if not start:
            if clan_tag == last:
                start = True
            continue

        save_progress(clan_tag)

        members = get_clan_members(clan_tag)

        for m in members:
            ptag = m["tag"]

            if ptag not in known_players:
                known_players.add(ptag)
                writer.writerow([ptag])

        print(f"✔ Clan traité : {clan_tag} ({len(members)} joueurs)")

    player_file.close()

    print("\n🎉 Extraction terminée ! Tous les tags joueurs sont enregistrés.")


if __name__ == "__main__":
    main()
