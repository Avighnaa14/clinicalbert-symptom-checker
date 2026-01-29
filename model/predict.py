import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "saved_model"
DISEASE_MASTER_PATH = "data/disease_master.json"
MAX_LENGTH = 128

# 🔐 Confidence thresholds (PERCENT)
MIN_CONFIDENCE = 40.0        # top-1 must be ≥ this
DISPLAY_CONFIDENCE = 20.0    # shown in top-3 list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load disease master
with open(DISEASE_MASTER_PATH, "r", encoding="utf-8") as f:
    DISEASE_MASTER = json.load(f)

# Load trained model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

def predict_top3(symptoms_text: str):
    """
    Returns threshold-filtered predictions (report-ready)
    """

    inputs = tokenizer(
        symptoms_text,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    top_probs, top_ids = torch.topk(probs, k=3)

    predictions = []
    for i in range(3):
        disease_id = int(top_ids[0][i].item())
        confidence = float(top_probs[0][i].item()) * 100

        if confidence >= DISPLAY_CONFIDENCE:
            predictions.append({
                "disease_id": disease_id,
                "disease_name": DISEASE_MASTER[str(disease_id)],
                "probability_percent": round(confidence, 2)
            })

    # Decide if top-1 is confident enough
    if not predictions or predictions[0]["probability_percent"] < MIN_CONFIDENCE:
        return {
            "status": "low_confidence",
            "message": "No disease prediction with sufficient confidence.",
            "predictions": []
        }

    return {
        "status": "ok",
        "predictions": predictions,
        "top_disease": predictions[0]
    }


# Manual test
if __name__ == "__main__":
    symptoms = "fever, cough, sore throat, body pain"
    result = predict_top3(symptoms)

    print("Symptoms:", symptoms)
    print("Result:")
    print(json.dumps(result, indent=2))
