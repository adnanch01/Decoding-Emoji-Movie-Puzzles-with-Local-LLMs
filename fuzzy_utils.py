from fuzzywuzzy import fuzz
from normalization import normalize_title

def fuzzy_score(pred: str, gold: str, aliases: list) -> float:
    """Return the highest fuzzy ratio between pred and gold+aliases."""
    p = normalize_title(pred)
    candidates = [normalize_title(gold)] + [normalize_title(a) for a in aliases]
    return max(fuzz.ratio(p, c) / 100.0 for c in candidates)

def pick_best(candidates, gold, aliases, threshold=0.8):
    """Return (best_pred, best_score, correct_bool)."""
    scored = [(c, fuzzy_score(c, gold, aliases)) for c in candidates]
    best_pred, best_score = max(scored, key=lambda x: x[1])
    correct = best_score >= threshold
    return best_pred, best_score, correct
