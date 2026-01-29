import json
SEVERITY_ACTION = {
    "normal": "Monitor symptoms.",
    "moderate": "Consult a doctor.",
    "high": "Seek medical care."
}

SEVERITY_COLOR = {
    "normal": "green",
    "moderate": "yellow",
    "high": "red"
}


RECOMMENDATION_PATH = "data/disease_recommendations.json"

with open(RECOMMENDATION_PATH, "r", encoding="utf-8") as f:
    RECOMMENDATIONS_LIST = json.load(f)

def get_recommendation(disease_id: int):
    """
    Fetch recommendation from LIST-based JSON
    """
    for item in RECOMMENDATIONS_LIST:
        if item.get("disease_id") == disease_id:
            return item.get("recommendation", {})

    return {
        "do": [],
        "dont": [],
        "home_remedies": [],
        "note": "No recommendation available."
    }
