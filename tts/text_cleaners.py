import re
import unicodedata

ZERO_WIDTH = ["\u200c", "\u200d", "\u2060", "\ufeff"]  # ZWNJ, ZWJ, word-joiner, BOM

def nepali_cleaners(text: str, add_sentence_end: bool = False) -> str:
    # 1) Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # 2) Replace smart quotes/dashes
    text = (text
            .replace("‘", "'").replace("’", "'")
            .replace("“", '"').replace("”", '"')
            .replace("–", "-").replace("—", "-"))

    # 3) Remove zero-width/non-printing characters
    for ch in ZERO_WIDTH:
        text = text.replace(ch, "")

    # 4) Lowercase ASCII letters only (leave Nepali intact)
    text = "".join(c.lower() if (c.isascii() and c.isalpha()) else c for c in text)

    # 5) Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 6) Normalize sentence-ending punctuation (optional)
    # Convert '.' to danda if you want consistent Nepali endings:
    # text = re.sub(r"\.(\s*)$", "।\\1", text)

    # Only append if requested and if it doesn't already end properly
    if add_sentence_end and text and not re.search(r"[।॥.?!]$", text):
        text += "।"

    return text