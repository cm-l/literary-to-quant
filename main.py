#!/usr/bin/env python3
"""
run_spatial_benchmarks.py

1. Loads config from .env (API key, dirs, schema, max tokens)
2. Recursively loads scenario prompts from SCENARIO_DIR/<CATEGORY>/*.json
3. Loads SCENE_SCHEMA from a separate JSON file
4. Queries multiple models via OpenRouter, enforcing that schema
5. Retries once on schema-validation failure (temp=0), logs metadata
6. Saves each result under OUTPUT_DIR/<model_alias>/<category>/<scenario_id>.json

This version sanitizes the API key to remove non-ASCII characters to avoid
UnicodeEncodeError in HTTP headers, and embeds the full JSON schema into the system prompt.
"""

import os
import sys
import json
import time
import logging
import re
import requests
from jsonschema import validate, ValidationError
from dotenv import load_dotenv

# ================ Load .env & Env Vars ================
load_dotenv()

raw_key = os.getenv("OPENROUTER_API_KEY")
if not raw_key:
    print("ERROR: Please set OPENROUTER_API_KEY in your .env", file=sys.stderr)
    sys.exit(1)

# sanitize key: strip out non-ASCII characters
OPENROUTER_API_KEY = raw_key.encode('ascii', 'ignore').decode('ascii')
if OPENROUTER_API_KEY != raw_key:
    logging.basicConfig(level=logging.WARNING)
    logging.warning(
        "Non-ASCII characters were removed from OPENROUTER_API_KEY. "
        "Ensure your key uses only ASCII."
    )

SCHEMA_FILE   = os.getenv("SCHEMA_FILE", "scene_schema.json")
SCENARIO_DIR  = os.getenv("SCENARIO_DIR", "scenarios")
OUTPUT_DIR    = os.getenv("OUTPUT_DIR", "outputs")
API_URL       = os.getenv("API_URL", "https://openrouter.ai/api/v1/chat/completions")
MAX_TOKENS    = int(os.getenv("MAX_TOKENS", "512"))

# ================ Models & Params ================
MODELS = {
    "gpt-4o":      "openai/gpt-4o",
    "claude-3.7":  "anthropic/claude-3.7-sonnet",
    "llama-3-70b": "meta-llama/llama-3-70b-instruct",
}

PARAMS = {
    "temperature": 0.2,
    "top_p":       0.95,
    "max_tokens":  MAX_TOKENS,
    "stop":        ["\n\n"],
}
FALLBACK_PARAMS = { **PARAMS, "temperature": 0.0 }

# ================ Logging Setup ================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ================ Load JSON Schema ================
try:
    with open(SCHEMA_FILE) as f:
        SCENE_SCHEMA = json.load(f)
    logging.info(f"Loaded JSON schema from {SCHEMA_FILE}")
    # Serialize schema for injection into system prompt
    SCHEMA_STR = json.dumps(SCENE_SCHEMA, indent=2)
    logging.info("Serialized scene schema for prompt injection")
except Exception as e:
    logging.error(f"Failed to load schema file '{SCHEMA_FILE}': {e}")
    sys.exit(1)

# ================ Helpers ================

def load_scenarios(base_path):
    """
    Load all scenarios from subfolders of base_path.
    Expected structure:
      base_path/
        CALIBRATION/*.json
        CARDINAL/*.json
        CONTEXT   /*.json
        LITERARY  /*.json
    Each JSON file must contain at least {"prompt": "..."}.
    """
    scenarios = []
    for category in os.listdir(base_path):
        cat_dir = os.path.join(base_path, category)
        if not os.path.isdir(cat_dir):
            continue
        for fname in os.listdir(cat_dir):
            if not fname.endswith(".json"):
                continue
            full_path = os.path.join(cat_dir, fname)
            data = json.load(open(full_path))
            scenarios.append({
                "id":       os.path.splitext(fname)[0],
                "prompt":   data["prompt"],
                "category": category
            })
    return scenarios


def call_model(model_slug, messages, params):
    """
    POST to OpenRouter and return:
      ok: bool
      message_content: str  ← JUST the assistant’s content, no wrappers
      metadata: dict
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":    model_slug,
        "messages": messages,
        **{k: v for k, v in params.items() if v is not None},
    }
    start = time.time()
    resp = requests.post(API_URL, headers=headers, json=payload)
    latency = time.time() - start

    meta = {
        "status_code": resp.status_code,
        "latency_s":   round(latency, 3),
        "usage":       resp.headers.get("x-usage-tokens"),
        "model_used":  resp.headers.get("x-router-model") or model_slug,
    }

    # HTTP error?
    if resp.status_code != 200:
        meta["response_text"] = resp.text[:500]
        logging.error(f"HTTP {resp.status_code}: {resp.text}")
        return False, resp.text, meta

    # Parse JSON
    try:
        data = resp.json()
    except ValueError as e:
        meta["response_text"] = resp.text[:500]
        logging.error(f"Invalid JSON: {e}")
        return False, resp.text, meta

    # Extract the assistant message
    choice = data["choices"][0]
    msg = choice["message"]["content"]
    meta["finish_reason"] = choice.get("finish_reason")

    # Strip ```json fences if present
    m = re.search(r"```(?:json)?\n([\s\S]*?)```", msg)
    if m:
        msg = m.group(1)

    return True, msg, meta


def validate_schema(text):
    """Parse text→JSON and validate against SCENE_SCHEMA."""
    obj = json.loads(text)
    validate(instance=obj, schema=SCENE_SCHEMA)
    return obj

# ================ Main Loop ================

def main():
    scenarios = load_scenarios(SCENARIO_DIR)
    logging.info(f"Loaded {len(scenarios)} scenarios from '{SCENARIO_DIR}'")

    for alias, slug in MODELS.items():
        for sc in scenarios:
            # outputs grouped by model and category
            out_dir = os.path.join(OUTPUT_DIR, alias, sc["category"])
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f"{sc['id']}.json")
            if os.path.exists(out_file):
                logging.info(f"Skipping existing: {out_file}")
                continue

            # Build the messages with embedded schema
            system_prompt = (
                "You are a spatial-reasoning agent. Arrange the objects mentioned in the text accordingly."
                "Use a right-handed XYZ frame, origin at room center. "
                "Answer ONLY with JSON that conforms exactly to the schema below—no extra keys, no commentary.\n\n"
                "Here is the JSON schema for scene descriptions:\n"
                "```json\n"
                f"{SCHEMA_STR}\n"
                "```"
            )
            messages = [
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": f"Scene: \"{sc['prompt']}\""}
            ]

            # First pass
            ok, text, meta = call_model(slug, messages, PARAMS)

            if ok:
                try:
                    # Validate against schema but do not save parsed output
                    validate_schema(text)
                    meta["retried"] = False
                except (json.JSONDecodeError, ValidationError):
                    logging.info(f"{alias}/{sc['id']}: schema validation failed, retrying")
                    ok2, text2, meta2 = call_model(slug, messages, FALLBACK_PARAMS)
                    meta.update({"retried": True, **meta2})
                    if ok2:
                        try:
                            validate_schema(text2)
                            text = text2
                        except Exception as e:
                            logging.error(f"{alias}/{sc['id']}: retry validation failed: {e}")
                    else:
                        logging.error(f"{alias}/{sc['id']}: retry HTTP error")
            else:
                logging.error(f"{alias}/{sc['id']}: initial request failed")

            # Save results (only raw_output)
            result = {
                "scenario_id":  sc["id"],
                "category":     sc["category"],
                "prompt":       sc["prompt"],
                "model":        alias,
                "parameters":   FALLBACK_PARAMS if meta.get("retried") else PARAMS,
                "llm_metadata": meta,
                "raw_output":   text,
            }
            with open(out_file, "w") as f:
                json.dump(result, f, indent=2)
            logging.info(f"Wrote: {out_file}")

if __name__ == "__main__":
    main()
