import pandas as pd
import re
import os

INPUT_FILE = "data/raw/symptoms_raw.csv"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "symptoms_cleaned.csv")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_symptoms(text):
    # Absolute safety check
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9,\s]", "", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df = pd.read_csv(INPUT_FILE)

# FORCE replace NaN with empty string BEFORE apply
df["symptoms_text"] = df["symptoms_text"].fillna("")

# Clean text
df["symptoms_text"] = df["symptoms_text"].apply(clean_symptoms)

# Keep only valid disease IDs
df = df[df["disease_id"].between(0, 53)]

# Remove empty / useless rows
df = df[df["symptoms_text"].str.len() > 10]

df.to_csv(OUTPUT_FILE, index=False)

print("✅ Cleaning successful")
print(f"Saved file → {OUTPUT_FILE}")
print(f"Total rows after cleaning: {len(df)}")
