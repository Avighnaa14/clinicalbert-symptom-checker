from datetime import datetime
from .predict import predict_top3
from .recommendation import (
    get_recommendation,
    SEVERITY_ACTION,
    SEVERITY_COLOR
)


def get_severity_from_symptoms(symptoms: str) -> str:
    symptom_list = [s for s in symptoms.split(",") if s.strip()]
    count = len(symptom_list)

    if count <= 2:
        return "normal"
    elif count <= 5:
        return "moderate"
    else:
        return "high"


def diagnose(symptoms: str):
    # Call prediction function
    result = predict_top3(symptoms)

    # 🔐 SAFELY extract predictions list
    # Handles both list and dict return types
    if isinstance(result, dict) and "predictions" in result:
        predictions = result["predictions"]
    else:
        predictions = result

    # Safety check (should never be empty, but defensive coding)
    if not predictions:
        return {
            "input_method": "Symptom Text",
            "input": symptoms,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_diseases": [],
            "overall_model_confidence_percent": 0.0,
            "severity_level": "normal",
            "severity_color": SEVERITY_COLOR["normal"],
            "severity_action": SEVERITY_ACTION["normal"],
            "disease_specific_recommendation": {}
        }

    # Top disease (rank 1)
    top_disease = predictions[0]

    # Severity based ONLY on symptom count
    severity = get_severity_from_symptoms(symptoms)

    # Overall model confidence = average of shown probabilities
    overall_confidence = round(
        sum(p["probability_percent"] for p in predictions) / len(predictions),
        2
    )

    return {
        "input_method": "Symptom Text",
        "input": symptoms,

        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "predicted_diseases": predictions,

        "overall_model_confidence_percent": overall_confidence,

        "severity_level": severity,
        "severity_color": SEVERITY_COLOR[severity],
        "severity_action": SEVERITY_ACTION[severity],

        "disease_specific_recommendation": get_recommendation(
            top_disease["disease_id"]
        )
    }
