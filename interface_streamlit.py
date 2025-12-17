import streamlit as st
import csv
import ast
import os
import sys

# importation logique
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from ia_predictive import analyse_combat
except ImportError:
    st.error("Erreur : Le fichier 'ia_predictive.py' est introuvable.")
    st.stop()

# configuration simple
st.set_page_config(
    page_title="Clash Royale AI - Interface Moderne",
    page_icon="👑",
    layout="wide"
)

# initialisation des variables de session (pour mémoriser les decks)
if "deck1" not in st.session_state:
    st.session_state.deck1 = []
if "deck2" not in st.session_state:
    st.session_state.deck2 = []

# gestion du clic (callback)
def toggle_selection(deck_key, card_name):
    current_list = st.session_state[deck_key]

    if card_name in current_list:
        current_list.remove(card_name)
    else:
        if len(current_list) < 8:
            current_list.append(card_name)
        else:
            st.toast("Le deck est déjà plein (8 cartes max) !", icon="⚠️")


# chargement des données
@st.cache_data
def load_card_data():
    path_images = "dataset/clashroyale_cards.csv"
    path_names = "dataset/cartes.csv"

    # charger le mapping Nom -> Image URL
    image_map = {}
    if os.path.exists(path_images):
        with open(path_images, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    raw_urls = row.get('iconUrls', "{}")
                    urls = ast.literal_eval(raw_urls)
                    if 'medium' in urls:
                        image_map[row.get('name')] = urls['medium']
                except:
                    continue

    # charger la liste officielle
    official_cards = []
    if os.path.exists(path_names):
        with open(path_names, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    name = row[1]
                    url = image_map.get(name, None)
                    official_cards.append({"name": name, "image_url": url})

    return official_cards


# chargement
cards_data = load_card_data()
name_to_url = {c["name"]: c["image_url"] for c in cards_data}


# fonction modale
@st.dialog("Sélectionner les cartes du deck")
def open_deck_selector(deck_key):
    # Compteur en temps réel
    count = len(st.session_state[deck_key])
    st.markdown(f"**Cartes sélectionnées : {count} / 8**")
    if count >= 8:
        st.info("Deck complet. Décochez une carte pour en changer.")

    # Barre de recherche
    search = st.text_input("🔍 Rechercher une carte...", key=f"search_{deck_key}")

    filtered_cards = [c for c in cards_data if search.lower() in c["name"].lower()]

    cols = st.columns(4)
    for i, card in enumerate(filtered_cards):
        c_name = card["name"]
        c_url = card["image_url"]
        is_selected = c_name in st.session_state[deck_key]

        with cols[i % 4]:
            # Affichage de l'image
            if c_url:
                st.image(c_url, use_container_width=True)
            else:
                st.write(c_name)

            # Bouton de sélection (Style dynamique)
            btn_label = "✅ Retirer" if is_selected else "Choisir"
            btn_type = "primary" if is_selected else "secondary"

            # Désactiver le bouton si le deck est plein et que la carte n'est pas celle sélectionnée
            is_disabled = (count >= 8 and not is_selected)

            # L'astuce est ici : on utilise on_click pour déclencher la mise à jour sans fermer la modale
            st.button(
                btn_label,
                key=f"btn_{deck_key}_{c_name}",
                type=btn_type,
                disabled=is_disabled,
                on_click=toggle_selection,
                args=(deck_key, c_name)
            )

    st.markdown("---")
    if st.button("Terminer la sélection", type="primary", use_container_width=True):
        st.rerun()



st.title("⚔️ Clash Royale - Analyseur de Deck IA")

# curseur précision analyse
precision = st.slider("Précision de l'analyse", 1, 8, 5)

st.markdown("---")

col1, col2 = st.columns(2, gap="large")

# deck 1
with col1:
    st.subheader("🔵 Deck 1")

    # btn ouvrir modale
    if st.button("Modifier Deck 1", use_container_width=True, icon="🃏"):
        open_deck_selector("deck1")

    st.markdown("<br>", unsafe_allow_html=True)

    # affichage cartes sélectionnées
    current_deck1 = st.session_state.deck1
    if current_deck1:
        d_cols = st.columns(4)
        for i, card_name in enumerate(current_deck1):
            img = name_to_url.get(card_name)
            with d_cols[i % 4]:
                if img:
                    st.image(img, caption=card_name, use_container_width=True)
                else:
                    st.info(card_name)
    else:
        st.info("Aucune carte sélectionnée.")

# deck 2
with col2:
    st.subheader("🔴 Deck 2")

    # btn ouvrir modale
    if st.button("Modifier Deck 2", use_container_width=True, icon="🃏"):
        open_deck_selector("deck2")

    st.markdown("<br>", unsafe_allow_html=True)

    # affichage cartes sélectionnées
    current_deck2 = st.session_state.deck2
    if current_deck2:
        d_cols = st.columns(4)
        for i, card_name in enumerate(current_deck2):
            img = name_to_url.get(card_name)
            with d_cols[i % 4]:
                if img:
                    st.image(img, caption=card_name, use_container_width=True)
                else:
                    st.info(card_name)
    else:
        st.info("Aucune carte sélectionnée.")

# bouton action
st.markdown("---")
st.markdown("<br>", unsafe_allow_html=True)

# centrer bouton analyse
_, center_col, _ = st.columns([1, 2, 1])

with center_col:
    analyze_btn = st.button("VOIR LE VAINQUEUR", type="primary", use_container_width=True)

if analyze_btn:
    d1 = st.session_state.deck1
    d2 = st.session_state.deck2

    if len(d1) != 8 or len(d2) != 8:
        st.warning(f"⚠️ Les decks doivent être complets (8 cartes).\nDeck 1: {len(d1)}/8 | Deck 2: {len(d2)}/8")
    else:
        with st.spinner("Analyse des matchs historiques en cours..."):
            combats, p1, p2 = analyse_combat(d1, d2, precision)

        st.success("Analyse terminée !")

        # Affichage résultats
        r1, r2, r3 = st.columns(3)
        r1.metric("Combats analysés", combats, border=True)
        r2.metric("Victoire Deck 1", f"{p1}%", border=True)
        r3.metric("Victoire Deck 2", f"{p2}%", border=True)

        if combats > 0:
            st.progress(p1 / 100)
            if p1 > p2:
                st.markdown(f"### 🏆 Le **Deck 1** l'emporte statistiquement !")
            elif p2 > p1:
                st.markdown(f"### 🏆 Le **Deck 2** l'emporte statistiquement !")
            else:
                st.markdown("### 🤝 Égalité parfaite.")
        else:
            st.warning("Aucun combat similaire trouvé. Essayez de réduire la précision.")