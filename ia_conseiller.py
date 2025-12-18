import ollama


def obtenir_conseil_coach(deck_user, deck_suggere):
    """Génère une explication textuelle via Mistral local"""
    prompt = f"""
    Tu es un coach expert pro de Clash Royale. 
    Deck actuel : {', '.join(deck_user)}
    Deck suggéré : {', '.join(deck_suggere)}

    Explique techniquement en 3 phrases pourquoi ce deck suggéré est meilleur. 
    Parle des synergies, de la défense ou de la win condition. Ton court et pro.
    """
    try:
        response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']
    except Exception:
        return "Le coach Mistral est indisponible. Vérifie qu'Ollama est lancé (`ollama run mistral`)."