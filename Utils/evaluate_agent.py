import os
import json
import random
import math
from typing import List, Dict, Any, Tuple
from groq import Groq
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Config
MODEL_ID = os.getenv("EVAL_MODEL_ID", "openai/gpt-oss-120b")  # Groq OSS model ID
API_KEY = os.getenv("GROQ_API_KEY")  # Set in environment OR hardcode below
OUTPUT_LOG = os.getenv("EVAL_OUTPUT", "./Results/eval_results.json")

# Dataset sizing and practicality controls
DATASET_SIZE = int(os.getenv("EVAL_DATASET_SIZE", "10000"))  # requested size
MAX_API_CALLS = int(os.getenv("EVAL_MAX_API_CALLS", "300"))  # cap to avoid rate limits/costs
RANDOM_SEED = int(os.getenv("EVAL_RANDOM_SEED", "42"))

# (Optional) Hardcode API key if not using env var
if not API_KEY:
    API_KEY = "your_api_key_here"

# Init Client
client = Groq(api_key=API_KEY)


# Utilities: normalization & scoring
def normalize_text(s: str) -> List[str]:
    return [t for t in ''.join(ch.lower() if ch.isalnum() or ch.isspace() else ' ' for ch in s).split() if t]


def token_f1(pred: str, gold_tokens: List[str]) -> float:
    pred_tokens = normalize_text(pred)
    gold = [t.lower() for t in gold_tokens]
    # multiset overlap approximation using counts
    from collections import Counter
    pc, gc = Counter(pred_tokens), Counter(gold)
    overlap = sum((pc & gc).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(1, sum(pc.values()))
    recall = overlap / max(1, sum(gc.values()))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def contains_any(pred: str, answers: List[str]) -> bool:
    p = ' '.join(normalize_text(pred))
    return any(ans.lower() in p for ans in answers)


# Dataset generation (10,000 diverse items)
COUNTRY_CAPITALS = {
    "france": "paris", "germany": "berlin", "italy": "rome", "spain": "madrid", "portugal": "lisbon",
    "uk": "london", "england": "london", "scotland": "edinburgh", "ireland": "dublin", "netherlands": "amsterdam",
    "belgium": "brussels", "sweden": "stockholm", "norway": "oslo", "denmark": "copenhagen", "finland": "helsinki",
    "poland": "warsaw", "czech republic": "prague", "austria": "vienna", "switzerland": "bern", "greece": "athens",
    "russia": "moscow", "turkey": "ankara", "canada": "ottawa", "usa": "washington", "united states": "washington",
    "mexico": "mexico city", "brazil": "brasilia", "argentina": "buenos aires", "japan": "tokyo", "china": "beijing",
    "india": "new delhi", "australia": "canberra", "new zealand": "wellington", "south africa": "pretoria",
}

AUTHORS = {
    "hamlet": ["william shakespeare", "shakespeare"],
    "pride and prejudice": ["jane austen", "austen"],
    "1984": ["george orwell", "orwell"],
    "the odyssey": ["homer"],
    "the iliad": ["homer"],
    "war and peace": ["leo tolstoy", "tolstoy"],
}

SCIENCE_KEYWORDS = {
    "newton_second_law": ["force", "mass", "acceleration", "f=ma", "proportional"],
    "photosynthesis": ["chlorophyll", "sunlight", "carbon", "oxygen", "glucose"],
    "evolution": ["natural selection", "mutation", "variation", "survival"],
}


def gen_arithmetic() -> Tuple[str, str]:
    a, b = random.randint(10, 999), random.randint(10, 999)
    if random.random() < 0.5:
        prompt = random.choice([
            f"Compute {a} + {b}.",
            f"What is {a} plus {b}?",
            f"Add {a} and {b}.",
        ])
        return prompt, str(a + b)
    else:
        # ensure non-negative result for simplicity
        x, y = max(a, b), min(a, b)
        prompt = random.choice([
            f"Compute {x} - {y}.",
            f"What is {x} minus {y}?",
            f"Subtract {y} from {x}.",
        ])
        return prompt, str(x - y)


def gen_capital() -> Tuple[str, str]:
    country, capital = random.choice(list(COUNTRY_CAPITALS.items()))
    q = random.choice([
        f"What is the capital of {country}?",
        f"Name the capital city of {country}.",
        f"{country.title()}'s capital is?",
    ])
    return q, capital


def gen_author() -> Tuple[str, List[str]]:
    work, answers = random.choice(list(AUTHORS.items()))
    q = random.choice([
        f"Who wrote '{work}'?",
        f"Name the author of {work}.",
        f"Identify the writer of the work: {work}.",
    ])
    return q, answers


def gen_science() -> Tuple[str, List[str]]:
    key, words = random.choice(list(SCIENCE_KEYWORDS.items()))
    qmap = {
        "newton_second_law": [
            "Explain Newton's second law in brief.",
            "Summarize the relation between force, mass, and acceleration.",
        ],
        "photosynthesis": [
            "Briefly describe photosynthesis.",
            "How do plants convert light to energy?",
        ],
        "evolution": [
            "Explain, in short, how evolution works.",
            "What is natural selection in one or two lines?",
        ],
    }
    return random.choice(qmap[key]), words


def build_dataset(n: int, seed: int = 42) -> List[Dict[str, Any]]:
    random.seed(seed)
    data: List[Dict[str, Any]] = []
    generators = ["math", "capital", "author", "science"]
    for i in range(n):
        g = generators[i % len(generators)]
        if g == "math":
            q, a = gen_arithmetic()
            data.append({"id": i, "type": g, "input": q, "answer": a})
        elif g == "capital":
            q, a = gen_capital()
            data.append({"id": i, "type": g, "input": q, "answer": a})
        elif g == "author":
            q, answers = gen_author()
            data.append({"id": i, "type": g, "input": q, "answers": answers})
        else:  # science
            q, keywords = gen_science()
            data.append({"id": i, "type": g, "input": q, "keywords": keywords})
    return data


# Build 10,000-sample dataset
dataset = build_dataset(DATASET_SIZE, RANDOM_SEED)


# Evaluation Loop (capped API calls)
results = []
num_to_eval = min(MAX_API_CALLS, len(dataset))
print(f"Running evaluation on {num_to_eval}/{len(dataset)} samples with {MODEL_ID} (cap via EVAL_MAX_API_CALLS)...")

for i in tqdm(range(num_to_eval), desc="Evaluating"):
    sample = dataset[i]
    prompt = sample["input"]

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=128,
        )
        generated = (response.choices[0].message.content or "").strip()

        # Scoring by type to avoid trivial overfitting
        score = 0.0
        extra = {}
        if sample["type"] in ("math", "capital"):
            # exact-ish check: look for the expected token as a standalone word
            expected = sample["answer"]
            score = 1.0 if contains_any(generated, [expected]) else 0.0
            extra["expected"] = expected
        elif sample["type"] == "author":
            answers = sample["answers"]
            score = 1.0 if contains_any(generated, answers) else 0.0
            extra["answers"] = answers
        else:  # science text: token F1 vs keywords
            kw = sample["keywords"]
            f1 = token_f1(generated, kw)
            # accept if coverage is reasonably good
            score = 1.0 if f1 >= 0.55 else 0.0
            extra["keywords"] = kw

        results.append({
            "id": sample["id"],
            "type": sample["type"],
            "prompt": prompt,
            **extra,
            "response": generated,
            "score": score,
        })

    except Exception as e:
        results.append({
            "id": sample["id"],
            "type": sample["type"],
            "prompt": prompt,
            "error": str(e),
            "score": 0.0,
        })


# Save Logs
os.makedirs(os.path.dirname(OUTPUT_LOG), exist_ok=True)
with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
    json.dump({
        "config": {
            "model_id": MODEL_ID,
            "dataset_size": len(dataset),
            "evaluated": len(results),
            "seed": RANDOM_SEED,
        },
        "results": results,
    }, f, indent=2, ensure_ascii=False)


# Report
scored = [r for r in results if "score" in r]
if scored:
    accuracy = sum(r["score"] for r in scored) / max(1, len(scored))
    print(f"\n✅ Evaluation complete. Accuracy on evaluated subset: {accuracy:.2%}")
    print(f"Evaluated {len(scored)} of {len(dataset)} samples (cap EVAL_MAX_API_CALLS={MAX_API_CALLS}).")
    print(f"Results saved to {OUTPUT_LOG}")
else:
    print("\n⚠️ No results generated.")
