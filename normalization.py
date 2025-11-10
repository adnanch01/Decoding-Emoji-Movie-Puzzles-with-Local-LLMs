import re
import unicodedata

def normalize_title(title: str) -> str:
    # to string, lowercase
    s = str(title).strip().lower()

    # normalize unicode
    s = unicodedata.normalize("NFKD", s)

    # remove punctuation
    s = re.sub(r"[^\w\s]", " ", s)

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # drop leading "the"
    if s.startswith("the "):
        s = s[4:]

    return s

ALIASES = {
    "spider-man": ["spiderman"],
    "harry potter and the sorcerer's stone": [
        "harry potter and the philosopher's stone",
        "harry potter 1",
    ],
    "e.t. the extra-terrestrial": ["et", "e.t.", "et the extra terrestrial"],
    "the lion king": [],
    "titanic": [],
    "the matrix": [],
    "jurassic park": [],
}

def matches_gold(pred: str, gold: str) -> bool:
    if not pred:
        return False
    p = normalize_title(pred)
    g = normalize_title(gold)
    if p == g:
        return True
    for alias in ALIASES.get(gold, []):
        if p == normalize_title(alias):
            return True
    return False
