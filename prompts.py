# ZERO-SHOT
def zero_shot_prompt(emoji: str) -> str:
    return f"Decode the following emoji sequence into a movie title:\n{emoji}\nAnswer only with the movie title."

# JSON-CONSTRAINED
def json_constrained_prompt(emoji: str) -> str:
    return (
        "Decode the following emoji sequence into a movie title. "
        "Return your answer strictly in JSON with the format {\"title\": \"...\"}.\n\n"
        f"Emojis: {emoji}"
    )

# FEW-SHOT (5 EXAMPLES)
def few_shot_prompt(emoji: str) -> str:
    examples = [
        ("🕷️🧑", "Spider-Man"),
        ("👸❄️", "Frozen"),
        ("🚢💔", "Titanic"),
        ("🦁👑", "The Lion King"),
        ("🧙💍", "The Lord of the Rings")
    ]
    fewshot_text = "\n".join([f"{e} → {a}" for e, a in examples])
    return (
        "You are decoding emojis into movie titles. "
        "Here are some examples:\n"
        f"{fewshot_text}\n\n"
        f"Now decode this one:\n{emoji}\nAnswer only with the movie title."
    )
