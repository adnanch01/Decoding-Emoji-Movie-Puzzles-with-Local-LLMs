import csv
import os
from ollama_client import query_ollama
from prompts import zero_shot_prompt, json_constrained_prompt, few_shot_prompt

def run_smoke_test():
    emoji = "🧑🕷️🏙️"  # Spider-Man
    model = "mistral:7b"
    temps = [0.0, 0.3, 0.7]
    prompt_fns = [zero_shot_prompt, json_constrained_prompt, few_shot_prompt]
    output_file = "smoke_results.csv"

    print(f"[INFO] Writing results to: {os.path.abspath(output_file)}")

    # Open CSV and ensure it writes after each iteration
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["emoji", "prompt_type", "temperature", "response", "latency"])
        f.flush()

        for temp in temps:
            print(f"\n=== Temperature: {temp} ===")
            for prompt_fn in prompt_fns:
                prompt_name = prompt_fn.__name__
                print(f"\n--- Running {prompt_name} ---")
                prompt = prompt_fn(emoji)

                try:
                    response, latency = query_ollama(model, prompt, temperature=temp)
                    print(f"Response: {response}")
                    print(f"Latency: {latency:.2f}s")

                    writer.writerow([emoji, prompt_name, temp, response, round(latency, 2)])
                    f.flush()
                except Exception as e:
                    print(f"Error during {prompt_name}: {e}")
                    writer.writerow([emoji, prompt_name, temp, f"ERROR: {e}", ""])
                    f.flush()

    # Verify file creation
    if os.path.exists(output_file):
        size = os.path.getsize(output_file)
        print(f"\n[INFO] Smoke test completed successfully.")
        print(f"[INFO] CSV file created: {output_file} ({size} bytes)")
    else:
        print(f"[ERROR] CSV file was not created. Please check write permissions.")

if __name__ == "__main__":
    run_smoke_test()
