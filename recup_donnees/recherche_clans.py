import requests
import csv
import itertools
import os
from time import time

def load_api_key(path="cle_api.txt"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print("❌ Fichier cle_api.txt introuvable.")
        exit()

API_KEY = load_api_key()
BASE_URL = "https://api.clashroyale.com/v1"

OUTPUT_FILE = "../dataset/clans_trouves.csv"
PROGRESS_FILE = "../progression/progress.txt"

ALPHABET = "0289PYLQGRJCUV"

session = requests.Session()
session.headers.update({
    "Accept": "application/json",
    "Authorization": f"Bearer {API_KEY}"
})

PROGRESS_SAVE_INTERVAL = 200


def check_clan_exists(tag):
    """Retourne True si le clan existe et a des membres."""
    url = f"{BASE_URL}/clans/{tag.replace('#', '%23')}"
    try:
        r = session.get(url, timeout=1.5)
        if r.status_code != 200:
            return False
        return len(r.json().get("memberList", [])) > 0
    except:
        return False


def load_progress():
    """Charge progression : (dernier_tag, longueur_en_cours)."""
    if not os.path.exists(PROGRESS_FILE):
        return None, 1
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return None, 1
        tag, length = content.split(";")
        return (tag if tag != "None" else None), int(length)
    except:
        return None, 1


def save_progress(tag, length):
    """Sauvegarde progression actuelle."""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        f.write(f"{tag};{length}")


def generate_tags(tag_length, start_from=None):
    """Génère les tags pour une longueur donnée."""
    started = start_from is None

    for combo in itertools.product(ALPHABET, repeat=tag_length):
        tag = "#" + "".join(combo)
        if not started:
            if tag == start_from:
                started = True
            continue
        yield tag


def ensure_csv():
    """Crée le fichier CSV s'il n'existe pas."""
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ClanTag", "Statut"])  # NOUVEL EN-TÊTE


def main():

    ensure_csv()

    last_tag, last_length = load_progress()
    print(f"➡️ Reprise : last_tag={last_tag}, longueur={last_length}")

    csv_file = open(OUTPUT_FILE, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)

    t0 = time()
    global_counter = 0

    for length in range(last_length, 10):
        print(f"\n📏 Longueur de tag = {length}")

        start_from = last_tag if length == last_length else None
        last_tag = None

        for tag in generate_tags(length, start_from=start_from):

            global_counter += 1

            if global_counter % 200 == 0:
                print(f"Test {tag}")

            if global_counter % PROGRESS_SAVE_INTERVAL == 0:
                save_progress(tag, length)

            # Vérification du clan
            is_valid = check_clan_exists(tag)

            # Écriture CSV obligatoire
            writer.writerow([tag, "valide" if is_valid else "non valide"])
            csv_file.flush()

            if is_valid:
                print(f"✅ Clan valide trouvé : {tag}")

        save_progress(None, length + 1)
        print(f"➡️ Fin longueur {length}, passage à {length + 1}")

    csv_file.close()
    print(f"\n🎉 Fin totale en {time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
