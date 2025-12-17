import tkinter as tk
from tkinter import ttk, messagebox
import csv

CARTES_FILE = "dataset/cartes.csv"
DONNEES_COMBATS = "dataset/combats_joueurs.csv"

#chargement de toute les cartes depuis le fichier cartes.csv
def load_cards():
    cards = []
    with open(CARTES_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            cid = int(row[0])
            name = row[1]
            cards.append((cid, name))
    return cards


# une analyse double sens avec une précision spécifié par l'utilisateur (allant de 1 à 8)
# ------------------------------------------------
def analyse_combat(deck1_names, deck2_names, precision):
    combats_selectionnes = 0
    vic_d1 = 0
    vic_d2 = 0

    with open(DONNEES_COMBATS, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            w_cards = row[2:10]   # 8 cartes vainqueur
            l_cards = row[12:20]  # 8 cartes perdant


            # vérification sens Deck1 -> gagnant
            if (len(set(deck1_names) & set(w_cards)) >= precision and
                len(set(deck2_names) & set(l_cards)) >= precision):
                vic_d1 += 1
                combats_selectionnes += 1

            # vérification sens Deck2 -> gagnant
            if (len(set(deck2_names) & set(w_cards)) >= precision and
                len(set(deck1_names) & set(l_cards)) >= precision):
                vic_d2 += 1
                combats_selectionnes += 1

    if combats_selectionnes == 0:
        return 0, 0, 0

    p_d1 = round((vic_d1 / combats_selectionnes) * 100, 2)
    p_d2 = round((vic_d2 / combats_selectionnes) * 100, 2)

    return combats_selectionnes, p_d1, p_d2

# ================================================================
# Interface Tkinter Modernisée
# ================================================================
class ClashApp:

    def __init__(self, root):
        self.root = root
        self.cards = load_cards()
        self.deck1_vars = []
        self.deck2_vars = []

        root.title("Analyse Deck Clash Royale - IA Précise")
        root.geometry("1000x750")
        root.configure(bg="#1e1e1e")

        self.build_ui()

    # ----------------------------------------------------------
    def build_ui(self):
        title = tk.Label(self.root, text="Analyseur Clash Royale", fg="white", bg="#1e1e1e",
                         font=("Arial", 22, "bold"))
        title.pack(pady=10)

        # Frame principale scrollable
        main_frame = tk.Frame(self.root, bg="#1e1e1e")
        main_frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(main_frame, bg="#1e1e1e", highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=scrollbar.set)

        self.table_frame = tk.Frame(canvas, bg="#1e1e1e")
        canvas_window = canvas.create_window((0, 0), window=self.table_frame, anchor="nw")

        self.table_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.bind(
            "<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width)
        )

        # TABLEAU
        headers = ["ID", "Carte", "Deck 1", "Deck 2"]
        for col, h in enumerate(headers):
            tk.Label(self.table_frame, text=h, fg="white", bg="#1e1e1e",
                     font=("Arial", 12, "bold")).grid(row=0, column=col, padx=10, pady=5)

        for i, (cid, name) in enumerate(self.cards):
            row = i + 1
            tk.Label(self.table_frame, text=str(cid), fg="white", bg="#1e1e1e",
                     font=("Arial", 11)).grid(row=row, column=0, padx=5)

            tk.Label(self.table_frame, text=name, fg="white", bg="#1e1e1e", width=40, anchor="w",
                     font=("Arial", 11)).grid(row=row, column=1, padx=5)

            var1 = tk.BooleanVar(value=False)
            cb1 = tk.Checkbutton(self.table_frame, variable=var1, bg="#1e1e1e",
                                 activebackground="#2c2c2c", command=self.update_selection)
            cb1.grid(row=row, column=2)
            self.deck1_vars.append(var1)

            var2 = tk.BooleanVar(value=False)
            cb2 = tk.Checkbutton(self.table_frame, variable=var2, bg="#1e1e1e",
                                 activebackground="#2c2c2c", command=self.update_selection)
            cb2.grid(row=row, column=3)
            self.deck2_vars.append(var2)

        # CHOIX PRECISION
        precision_frame = tk.Frame(self.root, bg="#1e1e1e")
        precision_frame.pack(pady=10)

        tk.Label(precision_frame, text="Précision de l'IA (1 à 8) :", fg="white", bg="#1e1e1e",
                 font=("Arial", 13)).pack(side="left", padx=5)

        self.precision_var = tk.IntVar(value=5)
        precision_box = ttk.Combobox(precision_frame, textvariable=self.precision_var, width=5,
                                     values=list(range(1, 9)))
        precision_box.pack(side="left")

        # RESULTATS
        result_frame = tk.LabelFrame(self.root, text="Résultats", bg="#1e1e1e", fg="white",
                                     font=("Arial", 12, "bold"), padx=10, pady=10)
        result_frame.pack(fill="x", padx=10, pady=10)

        self.result_label = tk.Label(result_frame, text="Sélectionnez 8 cartes pour chaque deck...",
                                     fg="white", bg="#1e1e1e", font=("Arial", 14))
        self.result_label.pack()

    # ----------------------------------------------------------
    def update_selection(self):
        # Limiter à 8 cartes
        if sum(v.get() for v in self.deck1_vars) > 8:
            messagebox.showwarning("Limite", "Deck 1 peut contenir 8 cartes max.")
            self.reset_excess(self.deck1_vars)
        if sum(v.get() for v in self.deck2_vars) > 8:
            messagebox.showwarning("Limite", "Deck 2 peut contenir 8 cartes max.")
            self.reset_excess(self.deck2_vars)

        if sum(v.get() for v in self.deck1_vars) == 8 and sum(v.get() for v in self.deck2_vars) == 8:
            self.run_ia()

    def reset_excess(self, deck_vars):
        count = 0
        for v in deck_vars:
            if v.get():
                count += 1
                if count > 8:
                    v.set(False)

    # ----------------------------------------------------------
    def run_ia(self):
        precision = self.precision_var.get()

        deck1_names = [self.cards[i][1] for i, v in enumerate(self.deck1_vars) if v.get()]
        deck2_names = [self.cards[i][1] for i, v in enumerate(self.deck2_vars) if v.get()]

        combats, p1, p2 = analyse_combat(deck1_names, deck2_names, precision)

        self.result_label.config(
            text=f"Combats détectés (comptés double sens) : {combats}\n"
                 f"Victoire Deck 1 : {p1}%\n"
                 f"Victoire Deck 2 : {p2}%"
        )

# ================================================================
# Lancement
# ================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = ClashApp(root)
    root.mainloop()
