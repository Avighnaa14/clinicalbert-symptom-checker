import csv
import random

# -------------------------------
# Synonyms / variations
# -------------------------------

INTENSITY = ["", "mild", "moderate", "severe", "persistent"]

# -------------------------------
# Expanded symptom pools (STRONG)
# -------------------------------

DISEASE_SYMPTOMS = {
    0: ["fatigue", "weakness", "pale skin", "shortness of breath", "dizziness", "rapid heartbeat"],
    1: ["frequent urination", "excessive thirst", "fatigue", "blurred vision", "slow wound healing"],
    2: ["headache", "dizziness", "chest pain", "shortness of breath", "nosebleeds"],
    3: ["chronic cough", "weight loss", "night sweats", "fever", "chest pain"],
    4: ["high fever", "joint pain", "muscle pain", "rash", "headache"],
    5: ["high fever", "severe headache", "joint pain", "skin rash", "bleeding gums"],
    6: ["fever", "abdominal pain", "loss of appetite", "weakness", "diarrhea"],
    7: ["loose stools", "abdominal cramps", "dehydration", "nausea"],
    8: ["weight loss", "fatigue", "poor appetite", "stunted growth"],
    9: ["watery diarrhea", "vomiting", "severe dehydration", "leg cramps"],
    10: ["thirst", "dry mouth", "reduced urination", "dark urine"],
    11: ["blurred vision", "difficulty seeing", "vision loss", "distorted vision"],
    12: ["red eyes", "itching", "eye discharge", "gritty sensation"],
    13: ["wheezing", "shortness of breath", "chest tightness", "coughing"],
    14: ["fever", "cough", "sore throat", "body pain", "chills"],
    15: ["fever", "chest pain", "difficulty breathing", "productive cough", "fatigue"],
    16: ["hard stools", "abdominal discomfort", "bloating"],
    17: ["facial pain", "nasal congestion", "headache", "post nasal drip"],
    18: ["vomiting", "diarrhea", "stomach cramps", "nausea"],
    19: ["itching", "skin redness", "ring shaped rash", "scaly skin"],
    20: ["upper abdominal pain", "fatigue", "liver discomfort", "weight gain"],
    21: ["joint pain", "stiffness", "swelling", "reduced mobility"],
    22: ["high body temperature", "dizziness", "confusion", "dry skin"],
    23: ["fatigue", "jaundice", "abdominal pain", "dark urine"],
    24: ["fatigue", "nausea", "jaundice", "loss of appetite"],
    25: ["abdominal swelling", "fatigue", "confusion", "easy bruising"],
    26: ["memory loss", "confusion", "difficulty thinking", "personality changes"],
    27: ["burning stomach pain", "nausea", "indigestion"],
    28: ["stomach pain", "bloating", "heartburn", "nausea"],
    29: ["tremors", "slow movement", "muscle stiffness", "balance problems"],
    30: ["breast lump", "breast pain", "skin changes", "nipple discharge"],
    31: ["pelvic pain", "abnormal bleeding", "pain during intercourse"],
    32: ["burning urination", "frequent urination", "pelvic pain"],
    33: ["fatigue", "joint pain", "skin rash", "fever"],
    34: ["bleeding gums", "joint pain", "weakness", "tooth loss"],
    35: ["itching", "circular rash", "scaly skin", "hair loss"],
    36: ["fever", "headache", "confusion", "vomiting"],
    37: ["fever", "rash", "joint pain", "red eyes"],
    38: ["fever", "chills", "muscle pain", "headache"],
    39: ["irregular periods", "weight gain", "acne", "excess hair growth"],
    40: ["weight gain", "fatigue", "cold intolerance", "dry skin"],
    41: ["weight loss", "heat intolerance", "palpitations", "tremors"],
    42: ["chronic cough", "shortness of breath", "fatigue", "wheezing"],
    43: ["pelvic pain", "painful periods", "heavy bleeding"],
    44: ["blurred vision", "cloudy vision", "night vision difficulty"],
    45: ["weight loss", "frequent infections", "fatigue", "night sweats"],
    46: ["mouth ulcers", "difficulty swallowing", "oral pain"],
    47: ["persistent abdominal pain", "loss of appetite", "weight loss"],
    48: ["jaundice", "abdominal swelling", "weight loss", "itching"],
    49: ["persistent cough", "chest pain", "weight loss", "shortness of breath"],
    50: ["anal pain", "bleeding during bowel movements", "itching"],
    51: ["itchy skin", "red patches", "dry skin", "scaling"],
    52: ["pimples", "oily skin", "skin inflammation", "blackheads"],
    53: ["yellowing of skin", "dark urine", "fatigue", "pale stools"]
}

# -------------------------------
# Dataset generation
# -------------------------------

OUTPUT_FILE = "data/raw/symptoms_raw.csv"
SAMPLES_PER_DISEASE = 120

rows = []

for disease_id, symptoms in DISEASE_SYMPTOMS.items():
    for _ in range(SAMPLES_PER_DISEASE):
        k = random.randint(3, min(6, len(symptoms)))
        chosen = random.sample(symptoms, k)

        final = []
        for s in chosen:
            if random.random() > 0.15:  # slight noise
                final.append(f"{random.choice(INTENSITY)} {s}".strip())

        symptom_text = ", ".join(final)
        rows.append([symptom_text, disease_id])

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["symptoms_text", "disease_id"])
    writer.writerows(rows)

print(f"Strong symptom-only dataset created: {len(rows)} rows → {OUTPUT_FILE}")
