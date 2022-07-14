import json
import os

def get_evol_param(type):
    if not os.path.exists(f"config/evolution/{type}.json"):
        raise NotImplementedError(f'{type} evolution is not implemented yet.')
    with open(f"config/evolution/{type}.json") as f:
        evol = json.load(f)
    result = {}
    result["type"] = type
    for key in evol:
        result[key] = evol[key]
    return result
