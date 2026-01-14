import re
import unicodedata

ZERO_WIDTH = ["\u200c", "\u200d", "\u2060", "\ufeff"]  # ZWNJ, ZWJ, word-joiner, BOM

def nepali_cleaners(text: str, add_sentence_end: bool = False) -> str:
    # 1) Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # 2) Character normalization (rare chars → common equivalents)
    text = text.replace("ॠ", "ऋ")
    text = text.replace("॥", "।")

    # 3) Replace smart quotes/dashes
    text = (text
            .replace("‘", "'").replace("’", "'")
            .replace("“", '"').replace("”", '"')
            .replace("–", "-").replace("—", "-"))

    # 4) Remove zero-width/non-printing characters
    for ch in ZERO_WIDTH:
        text = text.replace(ch, "")

    # 5) Lowercase ASCII letters only (leave Nepali intact)
    text = "".join(c.lower() if (c.isascii() and c.isalpha()) else c for c in text)

    # 6) Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 7) Only append danda if requested
    if add_sentence_end and text and not re.search(r"[।.?!]$", text):
        text += "।"

    return text