# Words-to-Worlds 
This is a dataset associated with the paper [[Work in Progress]].

## Contents:
Scenario descriptions are separated into categories: CALIBRATION, CARDINAL, CONTEXT and LITERARY.

| Scenario type     | Explicit geometry                      | Implied/world knowledge     |
| ----------------- | -------------------------------------- | --------------------------- |
| CALIBRATION       | Exact coordinatess & numerical offsets                 | None                        |
| CARDINAL            | Cardinal directions, “centre”, “above” | Low (room origin assumed)   |
| CONTEXT | Corners, facing, gaze, affordances                  | Medium (object affordances) |
| LITERARY          | Vivid prose, vague descriptions, metaphors,                      | High (scene conventions)    |


## .env template
Some variables need to be stored as environment variables. Once downloaded, you should make an .env file that looks similiar to this:
```
OPENROUTER_API_KEY=sk-[...]
SCHEMA_FILE=scene_schema.json
SCENARIO_DIR=scenarios
OUTPUT_DIR=outputs
API_URL=https://openrouter.ai/api/v1/chat/completions
```
