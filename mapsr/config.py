"""Locations of the inputs and outputs, from environment variables (defaults are relative to the working directory)."""
import json
import os
from functools import lru_cache

MAPS_JSON = os.environ.get("MAPSR_MAPS", "maps.json")
TEXTURES = os.environ.get("MAPSR_TEXTURES", "textures")
TILES = os.environ.get("MAPSR_TILES", "tiles")
PAIRS = os.environ.get("MAPSR_PAIRS", "pairs")
LAYER_API = os.environ.get("MAPSR_LAYER_API", "https://squadcalc.app/api/get/layer")


@lru_cache(maxsize=1)
def load_maps():
    """Map frames from `maps.json` (see scripts/export_maps_json.mjs): {slug: {"min": [x, y], "size": [w, h]}} in metres."""
    with open(MAPS_JSON) as handle:
        return json.load(handle)
