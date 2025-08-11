
import os, yaml
from typing import Dict, Any, Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_GLOSSARY_PATH = os.path.join(DATA_DIR, "domain_glossary.yaml")

_cached: Dict[str, Any] | None = None

def load_glossary(path: str | None = None) -> Dict[str, Any]:
    global _cached
    path = path or DEFAULT_GLOSSARY_PATH
    if _cached is not None:
        return _cached
    if not os.path.exists(path):
        return {"version":0,"namespace":"default","fields":{},"business_rules":{}}
    with open(path, "r", encoding="utf-8") as f:
        _cached = yaml.safe_load(f) or {}
    return _cached

def get_field_doc(field: str) -> Optional[str]:
    g = load_glossary()
    fields = g.get("fields", {})
    info = fields.get(field)
    if not info:
        # try alias permutations
        alt = field.replace(" ", "").replace("-", "").replace("_", "")
        for k,v in fields.items():
            key_norm = k.replace(" ", "").replace("-", "").replace("_", "")
            if key_norm.lower() == alt.lower():
                info = v; break
    if not info:
        return None
    title = info.get("title", field)
    desc = info.get("description", "")
    examples = info.get("examples", [])
    return f"{title}: {desc}" + (f" Examples: {examples}" if examples else "")

def get_business_rules() -> Dict[str, str]:
    g = load_glossary()
    return g.get("business_rules", {})
