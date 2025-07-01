#!/usr/bin/env python3
"""
main.py

1. Loads config from .env (API key, dirs, schema, max tokens, optional seed)
2. Recursively loads scenario prompts from SCENARIO_DIR/<CATEGORY>/*.json
3. Loads SCENE_SCHEMA from a separate JSON file
4. Queries multiple models via OpenRouter, enforcing that schema
5. Retries once on schema-validation failure (temp=0), logs metadata
6. Saves each result under OUTPUT_DIR/<model_alias>/<category>/<scenario_id>.json

Important to note: the JSON schema is also embedded into the system prompt for better results and fewer retries, though
it can be removed (but your results may differ from the ones in our paper).
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

# Logging setup: INFO-level general, DEBUG for detailed trace
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

raw_key = os.getenv("OPENROUTER_API_KEY")
if not raw_key:
    logger.error("OPENROUTER_API_KEY not set in .env. Exiting.")
    sys.exit(1)

# sanitize key: strip out non-ASCII characters
OPENROUTER_API_KEY = raw_key.encode('ascii', 'ignore').decode('ascii')
if OPENROUTER_API_KEY != raw_key:
    logger.warning(
        "Non-ASCII characters removed from OPENROUTER_API_KEY. Ensure key is ASCII."
    )
else:
    logger.debug("API key loaded and sanitized (no changes).")

SCHEMA_FILE  = os.getenv("SCHEMA_FILE", "scene_schema.json")
SCENARIO_DIR = os.getenv("SCENARIO_DIR", "scenarios")
OUTPUT_DIR   = os.getenv("OUTPUT_DIR", "outputs")
API_URL      = os.getenv(
    "API_URL", "https://openrouter.ai/api/v1/chat/completions"
)
MAX_TOKENS   = int(os.getenv("MAX_TOKENS", "512"))

# Optional seed for reproducible outputs
raw_seed = os.getenv("SEED")
SEED = int(raw_seed) if raw_seed and raw_seed.isdigit() else None
if SEED is not None:
    logger.info(f"Using fixed seed={SEED} for reproducible responses")
else:
    logger.debug("No seed provided; responses may vary run to run.")

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
    "seed":        SEED,
}
FALLBACK_PARAMS = {**PARAMS, "temperature": 0.0}
logger.debug(f"Default params: {PARAMS}")
logger.debug(f"Fallback params: {FALLBACK_PARAMS}")

# ================ Load JSON Schema ================
try:
    with open(SCHEMA_FILE) as f:
        SCENE_SCHEMA = json.load(f)
    SCHEMA_STR = json.dumps(SCENE_SCHEMA, indent=2)
    logger.info(f"Loaded and serialized JSON schema from '{SCHEMA_FILE}'")
except Exception as e:
    logger.error(f"Failed to load JSON schema '{SCHEMA_FILE}': {e}")
    sys.exit(1)

# ================ Helpers ================

def load_scenarios(base_path):
    """
    Load all scenarios from subfolders of base_path. Each JSON must contain {"prompt"}.
    """
    scenarios = []
    for category in os.listdir(base_path):
        cat_dir = os.path.join(base_path, category)
        if not os.path.isdir(cat_dir):
            logger.debug(f"Skipping non-directory: {cat_dir}")
            continue
        for fname in os.listdir(cat_dir):
            if not fname.endswith(".json"): continue
            path = os.path.join(cat_dir, fname)
            try:
                data = json.load(open(path))
                scenarios.append({
                    "id":       os.path.splitext(fname)[0],
                    "prompt":   data.get("prompt", ""),
                    "category": category
                })
                logger.debug(f"Loaded scenario: {category}/{fname}")
            except Exception as e:
                logger.warning(f"Failed to read scenario '{path}': {e}")
    return scenarios


def call_model(model_slug, messages, params):
    """
    POST to OpenRouter, returning (ok, content, metadata).
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {"model": model_slug, "messages": messages}
    payload.update({k: v for k, v in params.items() if v is not None})
    logger.info(f"Sending request to {model_slug}")
    logger.debug(f"Request payload keys: {list(payload.keys())}")
    start = time.time()

    try:
        resp = requests.post(API_URL, headers=headers, json=payload)
    except Exception as e:
        logger.error(f"HTTP request failed for {model_slug}: {e}")
        return False, "", {"status_code": None, "error": str(e)}

    latency = time.time() - start
    meta = {
        "status_code": resp.status_code,
        "latency_s":   round(latency, 3),
        "usage":       resp.headers.get("x-usage-tokens"),
        "model_used":  resp.headers.get("x-router-model") or model_slug,
    }
    logger.info(
        f"Response from {model_slug}: status {meta['status_code']}, "
        f"latency {meta['latency_s']}s, usage {meta['usage']}"
    )

    if resp.status_code != 200:
        text_snippet = resp.text[:200].replace("\n", " ")
        logger.error(f"Error response text (truncated): {text_snippet}")
        meta["response_text_snippet"] = text_snippet
        return False, "", meta

    try:
        data = resp.json()
    except ValueError as e:
        snippet = resp.text[:200].replace("\n", " ")
        logger.error(f"Invalid JSON from {model_slug}: {e}; snippet: {snippet}")
        meta["response_text_snippet"] = snippet
        return False, "", meta

    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {}).get("content", "")
    meta["finish_reason"] = choice.get("finish_reason")
    logger.debug(f"Finish reason: {meta['finish_reason']}")

    # Strip ```json fences if present
    m = re.search(r"```(?:json)?\n([\s\S]*?)```", msg)
    if m:
        msg = m.group(1)
        logger.debug("Stripped JSON fences from response content.")

    return True, msg, meta


def validate_schema(text):
    obj = json.loads(text)
    validate(instance=obj, schema=SCENE_SCHEMA)
    return obj

# ================ Main Loop ================
def main():
    logger.info("Starting spatial benchmarks run.")
    scenarios = load_scenarios(SCENARIO_DIR)
    logger.info(f"Loaded {len(scenarios)} scenarios from '{SCENARIO_DIR}'")

    for alias, slug in MODELS.items():
        logger.info(f"Beginning runs for model '{alias}' ({slug})")
        for sc in scenarios:
            sc_id = sc["id"]
            logger.info(f"Processing scenario '{sc_id}' in category '{sc['category']}'")

            out_dir = os.path.join(OUTPUT_DIR, alias, sc['category'])
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f"{sc_id}.json")
            if os.path.exists(out_file):
                logger.info(f"Output exists, skipping: {out_file}")
                continue

            system_prompt = (
                "You are a spatial-reasoning agent. Use a right-handed XYZ frame, origin at room center. "
                "Answer *only* with JSON that conforms exactly to the schema below—no extra keys, no commentary.\n\n"
                "Here is the JSON schema for scene descriptions:\n```json\n"
                f"{SCHEMA_STR}\n```"
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Scene: \"{sc['prompt']}\""}
            ]

            ok, text, meta = call_model(slug, messages, PARAMS)
            if ok:
                try:
                    validate_schema(text)
                    meta["retried"] = False
                    logger.info(f"Schema validation passed for {alias}/{sc_id}")
                except (json.JSONDecodeError, ValidationError) as e:
                    logger.warning(
                        f"Validation failed for {alias}/{sc_id}: {e}. Retrying with fallback."
                    )
                    ok2, text2, meta2 = call_model(slug, messages, FALLBACK_PARAMS)
                    meta.update({"retried": True, **meta2})
                    if ok2:
                        try:
                            validate_schema(text2)
                            text = text2
                            logger.info(
                                f"Retry validation passed for {alias}/{sc_id}"
                            )
                        except Exception as ex:
                            logger.error(
                                f"Retry validation failed for {alias}/{sc_id}: {ex}"
                            )
                    else:
                        logger.error(
                            f"Retry HTTP error for {alias}/{sc_id}: "
                            f"status {meta2.get('status_code')}"
                        )
            else:
                logger.error(
                    f"Initial request failed for {alias}/{sc_id}: "
                    f"status {meta.get('status_code')}"
                )

            # Save only raw_output
            result = {
                "scenario_id": sc_id,
                "category":    sc['category'],
                "prompt":      sc['prompt'],
                "model":       alias,
                "parameters":  FALLBACK_PARAMS if meta.get("retried") else PARAMS,
                "llm_metadata":meta,
                "raw_output":  text,
            }
            with open(out_file, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Wrote result to {out_file}")

    logger.info("Spatial benchmarks run complete.")

if __name__ == "__main__":
    main()
