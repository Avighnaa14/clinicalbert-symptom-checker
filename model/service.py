from .predict import predict_top3
from .recommendation import get_recommendation


def diagnose(symptoms_text: str):
    """
    Full diagnosis pipeline:
    - Step 7: Prediction
    - Step 8: Recommendation
    """

    prediction = predict_top3(symptoms_text)

    # Low confidence → stop here
    if prediction["status"] != "ok":
        return {
            "status": "low_confidence",
            "message": prediction["message"],
            "predictions": []
        }

    top_disease = prediction["top_disease"]
    disease_id = top_disease["disease_id"]

    recommendation = get_recommendation(disease_id)

    return {
        "status": "ok",
        "input_symptoms": symptoms_text,
        "predictions": prediction["predictions"],
        "top_disease": top_disease,
        "recommendation": recommendation
    }
