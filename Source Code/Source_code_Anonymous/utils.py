def load_html_snippet(path: str) -> str:
    """
    Legge e restituisce il contenuto di un file HTML.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()  # legge l'intero file in una stringa :contentReference[oaicite:5]{index=5}
