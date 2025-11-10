import json
import csv
import pandas as pd
import numpy as np
from ollama_client import query_ollama
from normalization import normalize_title, matches_gold
from prompts import zero_shot_prompt, json_constrained_prompt, few_shot_prompt


MODELS = ["mistral:7b", "llama3.2:latest"]  
TEMPS = [0.0, 0.3, 0.7]
PROMPTS = [
    ("zero_shot", zero_shot_prompt),
    ("json_constrained", json_constrained_prompt),
    ("few_shot", few_shot_prompt),
]


# Load emoji puzzles

with open("emoji_puzzles.json", "r", encoding="utf-8") as f:
    PUZZLES = json.load(f)

print(f"[INFO] Loaded {len(PUZZLES)} emoji puzzles.")



# Extract title helper (handles JSON prompt outputs)

def extract_title(prompt_name: str, raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return ""

    if prompt_name == "json_constrained":
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "title" in data:
                return data["title"].strip()
        except json.JSONDecodeError:
            # fallback: try to extract JSON substring
            if "{" in raw and "}" in raw:
                try:
                    obj_str = raw[raw.index("{"): raw.rindex("}") + 1]
                    data = json.loads(obj_str)
                    return str(data.get("title", "")).strip()
                except Exception:
                    return ""
        return ""
    else:
        # zero-shot / few-shot: take first line as guess
        return raw.split("\n")[0].strip()



# Main experiment loop

def run_grid():
    results = []
    for item in PUZZLES:
        emoji = item["emoji"]
        gold = item["answer"]
        for model in MODELS:
            for temp in TEMPS:
                for prompt_name, prompt_fn in PROMPTS:
                    prompt = prompt_fn(emoji)
                    try:
                        response, latency = query_ollama(model, prompt, temperature=temp)
                        pred = extract_title(prompt_name, response)
                        correct = matches_gold(pred, gold)
                    except Exception as e:
                        response = f"ERROR: {e}"
                        pred = ""
                        correct = False
                        latency = -1

                    results.append({
                        "model": model,
                        "emoji": emoji,
                        "gold": gold,
                        "prompt_type": prompt_name,
                        "temperature": temp,
                        "prediction": pred,
                        "correct": int(correct),
                        "latency": latency,
                    })

                    print(
                        f"[{model} | {prompt_name} | T={temp}] "
                        f"{emoji} → {pred} | Correct: {correct} | {latency:.2f}s"
                    )

    return results

# Save results to CSV

def save_results(results):
    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"[INFO] Saved {len(results)} rows to results.csv")



# Summarize accuracy + latency by condition

def summarize():
    df = pd.read_csv("results.csv")
    df = df[df["latency"] >= 0]

    summary = (
        df.groupby(["model", "prompt_type", "temperature"])
          .agg(
              accuracy=("correct", "mean"),
              mean_latency=("latency", "mean"),
              p95_latency=("latency", lambda x: np.percentile(x, 95))
          )
          .reset_index()
    )

    summary["accuracy"] = (summary["accuracy"] * 100).round(1)
    summary["mean_latency"] = summary["mean_latency"].round(2)
    summary["p95_latency"] = summary["p95_latency"].round(2)

    summary.to_csv("summary.csv", index=False)
    print("\n=== Summary (Accuracy & Latency by Model) ===")
    print(summary)
    print("\n[INFO] Saved summary to summary.csv")


# Run everything

if __name__ == "__main__":
    results = run_grid()
    save_results(results)
    summarize()
