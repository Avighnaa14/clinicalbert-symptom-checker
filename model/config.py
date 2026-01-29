MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

NUM_LABELS = 54            # diseases 0–53
MAX_LENGTH = 128           # enough for symptom-only text
BATCH_SIZE = 8             # safe for CPU / low GPU
LEARNING_RATE = 2e-5
EPOCHS = 5
RANDOM_SEED = 42
