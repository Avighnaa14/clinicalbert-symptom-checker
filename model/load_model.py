from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MODEL_NAME, NUM_LABELS

def load_tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )

    return tokenizer, model

if __name__ == "__main__":
    tokenizer, model = load_tokenizer_and_model()
    print("✅ ClinicalBERT tokenizer and model loaded successfully")
    print(f"Number of labels: {model.config.num_labels}")
