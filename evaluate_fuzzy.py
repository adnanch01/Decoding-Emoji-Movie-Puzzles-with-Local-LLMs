import json, time
from ollama_client import query_ollama
from prompts import zero_shot_prompt
from fuzzy_utils import pick_best
from normalization import ALIASES  # your alias dict

MODEL = "mistral:7b"
N = 5            # number of candidates
TEMP = 1.0       # higher for diversity
THRESH = 0.8

with open("emoji_puzzles.json", "r", encoding="utf-8") as f:
    PUZZLES = json.load(f)

results = []

for item in PUZZLES:
    emoji = item["emoji"]
    gold = item["answer"]
    aliases = item.get("aliases", [])
    prompt = zero_shot_prompt(emoji)

    candidates, total_latency = [], 0
    for _ in range(N):
        response, latency = query_ollama(MODEL, prompt, temperature=TEMP)
        candidates.append(response.strip())
        total_latency += latency

    best_pred, best_score, correct = pick_best(candidates, gold, aliases, threshold=THRESH)
    avg_latency = total_latency / N
    results.append((emoji, gold, best_pred, best_score, correct, avg_latency))
    print(f"{emoji} → {best_pred} | score={best_score:.2f} | correct={correct} | {avg_latency:.2f}s")

# optional: save results
import csv
with open("fuzzy_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["emoji", "gold", "best_pred", "score", "correct", "avg_latency"])
    writer.writerows(results)
print("[INFO] Saved fuzzy_results.csv")
