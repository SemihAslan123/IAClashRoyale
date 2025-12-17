# Nouvelle interface moderne avec deck choisi + deck optimal + précision + scroll molette

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import csv
import os
from collections import Counter

CARTES_FILE = "dataset/cartes.csv"
IMAGES_FOLDER = "images_cartes/"
DONNEES_COMBATS = "dataset/combats_joueurs.csv"


# Charger cartes
def load_cards():
    cards = []
    with open(CARTES_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            cid = int(row[0])
            name = row[1]
            cards.append((cid, name))
    return cards


# IA générative
def generer_deck_anti(deck_user, precision):
    compteur = Counter()
    combats = 0

    with open(DONNEES_COMBATS, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            gagnant = row[2:10]
            perdant = row[12:20]

            if len(set(deck_user) & set(perdant)) >= precision:
                compteur.update(gagnant)
                combats += 1

            if len(set(deck_user) & set(gagnant)) >= precision:
                compteur.update(perdant)
                combats += 1

    if combats == 0:
        return [], 0

    return [c for c, _ in compteur.most_common(8)], combats


class IA_Generative_App:

    def __init__(self, root):
        self.root = root
        self.cards = load_cards()
        self.selected_cards = []
        self.card_widgets = []
        self.images_cache = {}

        root.title("IA Générative Clash Royale - Interface Moderne")
        root.geometry("1200x800")
        root.configure(bg="#121212")

        self.build_ui()

    def build_ui(self):
        title = tk.Label(self.root, text="Sélectionne ton deck (8 cartes)", fg="white",
                         bg="#121212", font=("Arial", 22, "bold"))
        title.pack(pady=10)

        # Sélecteur précision
        p_frame = tk.Frame(self.root, bg="#121212")
        p_frame.pack(pady=5)
        tk.Label(p_frame, text="Précision :", fg="white", bg="#121212", font=("Arial", 14)).pack(side="left")

        self.precision_var = tk.IntVar(value=5)
        self.precision_box = ttk.Combobox(p_frame, textvariable=self.precision_var,
                                          values=list(range(1, 9)), width=5)
        self.precision_box.pack(side="left", padx=5)

        # Scroll Frame
        container = tk.Frame(self.root)
        container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(container, bg="#121212", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.grid_frame = tk.Frame(self.canvas, bg="#121212")
        self.canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")

        self.grid_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Molette de souris
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.display_cards_grid()

        # --- NOUVEAU : zone deck choisi ---
        self.deck_frame = tk.Frame(self.root, bg="#121212")
        self.deck_frame.pack(pady=10)

        self.deck_title = tk.Label(self.deck_frame, text="", fg="#00bfff", bg="#121212", font=("Arial", 16, "bold"))
        self.deck_title.pack()

        self.deck_cards_frame = tk.Frame(self.deck_frame, bg="#121212")
        self.deck_cards_frame.pack()

        # --- NOUVEAU : zone deck optimal ---
        self.optimal_frame = tk.Frame(self.root, bg="#121212")
        self.optimal_frame.pack(pady=10)

        self.optimal_title = tk.Label(self.optimal_frame, text="", fg="orange", bg="#121212",
                                      font=("Arial", 16, "bold"))
        self.optimal_title.pack()

        self.optimal_cards_frame = tk.Frame(self.optimal_frame, bg="#121212")
        self.optimal_cards_frame.pack()

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def load_card_image(self, card_name, size=120):
        img_path = os.path.join(IMAGES_FOLDER, f"{card_name}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(IMAGES_FOLDER, "vide.png")

        key = f"{img_path}_{size}"
        if key not in self.images_cache:
            img = Image.open(img_path).resize((size, size))
            self.images_cache[key] = ImageTk.PhotoImage(img)

        return self.images_cache[key]

    def display_cards_grid(self):
        cols = 5
        for idx, (cid, name) in enumerate(self.cards):
            row = idx // cols
            col = idx % cols

            frame = tk.Frame(self.grid_frame, bg="#1e1e1e", padx=5, pady=5,
                             highlightthickness=2, highlightbackground="#333",
                             width=150, height=180)
            frame.grid(row=row, column=col, padx=12, pady=12)
            frame.pack_propagate(False)

            img = self.load_card_image(name)
            img_label = tk.Label(frame, image=img, bg="#1e1e1e")
            img_label.pack()

            name_label = tk.Label(frame, text=name, fg="white", bg="#1e1e1e", font=("Arial", 11))
            name_label.pack(pady=4)

            for widget in (frame, img_label, name_label):
                widget.bind("<Button-1>", lambda e, n=name, w=frame: self.toggle_card(n, w))

            self.card_widgets.append((name, frame))

    def toggle_card(self, card_name, widget):
        if card_name in self.selected_cards:
            self.selected_cards.remove(card_name)
            widget.config(highlightbackground="#333")
        else:
            if len(self.selected_cards) >= 8:
                messagebox.showwarning("Limite atteinte", "Tu ne peux sélectionner que 8 cartes.")
                return
            self.selected_cards.append(card_name)
            widget.config(highlightbackground="#00bfff")

        self.update_deck_display()

        if len(self.selected_cards) == 8:
            self.run_ia()

    # --- NOUVEAU : affichage du deck choisi ---
    def update_deck_display(self):
        for w in self.deck_cards_frame.winfo_children():
            w.destroy()

        if len(self.selected_cards) == 0:
            self.deck_title.config(text="")
            return

        self.deck_title.config(text=f"Deck sélectionné : {len(self.selected_cards)}/8 cartes")

        for name in self.selected_cards:
            frame = tk.Frame(self.deck_cards_frame, bg="#1e1e1e", padx=3, pady=3)
            frame.pack(side="left", padx=5)

            img = self.load_card_image(name, size=60)
            tk.Label(frame, image=img, bg="#1e1e1e").pack()
            tk.Label(frame, text=name, fg="white", bg="#1e1e1e", font=("Arial", 10)).pack()

    # --- NOUVEAU : affichage du deck optimal ---
    def display_optimal_deck(self, deck):
        for w in self.optimal_cards_frame.winfo_children():
            w.destroy()

        if not deck:
            self.optimal_title.config(text="")
            return

        self.optimal_title.config(text="Deck optimal contre ton deck")

        for name in deck:
            frame = tk.Frame(self.optimal_cards_frame, bg="#1e1e1e", padx=3, pady=3)
            frame.pack(side="left", padx=5)

            img = self.load_card_image(name, size=60)
            tk.Label(frame, image=img, bg="#1e1e1e").pack()
            tk.Label(frame, text=name, fg="white", bg="#1e1e1e", font=("Arial", 10)).pack()

    def run_ia(self):
        precision = self.precision_var.get()
        suggested, combats = generer_deck_anti(self.selected_cards, precision)

        if combats == 0:
            self.optimal_title.config(text="Aucun combat correspondant trouvé.")
            return

        self.display_optimal_deck(suggested)


if __name__ == "__main__":
    root = tk.Tk()
    IA_Generative_App(root)
    root.mainloop()
