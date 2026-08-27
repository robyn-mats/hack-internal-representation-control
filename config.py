import os

PROFILES = {
    "dev": {
        "model_id":    "google/gemma-3-4b-it",
        "sae_release": "gemma-scope-2-4b-it-res",
        "layer":       22,
        "saes": {
            "pilot": "layer_22_width_16k_l0_medium",
            "main":  "layer_22_width_262k_l0_medium",
            "seed1": "layer_22_width_262k_l0_medium_seed_1",
        },
    },
    "prod": {
        "model_id":    "google/gemma-3-27b-it",
        "sae_release": "gemma-scope-2-27b-it-res",
        "layer":       40,
        "saes": {
            "pilot": "layer_40_width_16k_l0_medium",
            "main":  "layer_40_width_262k_l0_medium",
            "seed1": "layer_40_width_262k_l0_medium_seed_1",
        },
    },
}

MODE = os.environ.get("WB_MODE", "dev")
assert MODE in PROFILES, f"WB_MODE={MODE!r} not in {list(PROFILES)}"

_p          = PROFILES[MODE]
MODEL_ID    = _p["model_id"]
SAE_RELEASE = _p["sae_release"]
LAYER       = _p["layer"]
SAES        = _p["saes"]
SAE_ID      = SAES["pilot"]          # default readout

EXTRA_MODELS = [
    "google/gemma-3-12b-it",   # capacity fallback
    "Qwen/Qwen3.5-4B",         # J-lens walkthrough
]

def stamp():
    """Provenance dict — write into every output file."""
    return {"mode": MODE, "model_id": MODEL_ID, "sae_release": SAE_RELEASE,
            "layer": LAYER, "sae_id": SAE_ID}