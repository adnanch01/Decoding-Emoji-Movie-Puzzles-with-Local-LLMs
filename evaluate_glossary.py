import json, pandas as pd
from ollama_client import query_ollama
from normalization import matches_gold
from prompts import rag_glossary_prompt

MODEL = "mistral:7b"
TEMP = 0.3

with open("emoji_puzzles.json", "r", encoding="utf-8") as f:
    PUZZLES = json.load(f)

results = []

for item in PUZZLES:
    emoji = item["emoji"]
    gold = item["answer"]
    prompt = rag_glossary_prompt(emoji)
    response, latency = query_ollama(MODEL, prompt, temperature=TEMP)
    correct = matches_gold(response, gold)
    results.append({"emoji": emoji, "gold": gold, "pred": response, "correct": correct, "latency": latency})
    print(f"{emoji} → {response} | correct={correct} | {latency:.2f}s")

df = pd.DataFrame(results)
df.to_csv("rag_glossary_results.csv", index=False)
print("\nAccuracy:", df["correct"].mean() * 100, "%")
print("Mean latency:", df["latency"].mean(), "s")
