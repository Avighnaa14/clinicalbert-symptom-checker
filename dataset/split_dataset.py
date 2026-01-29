import pandas as pd
from sklearn.model_selection import train_test_split
import os

INPUT_FILE = "data/processed/symptoms_cleaned.csv"
OUTPUT_DIR = "data/processed"

TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.csv")
VAL_FILE = os.path.join(OUTPUT_DIR, "val.csv")
TEST_FILE = os.path.join(OUTPUT_DIR, "test.csv")

df = pd.read_csv(INPUT_FILE)

# First split: Train (80%) + Temp (20%)
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["disease_id"],
    random_state=42
)

# Second split: Validation (10%) + Test (10%)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["disease_id"],
    random_state=42
)

train_df.to_csv(TRAIN_FILE, index=False)
val_df.to_csv(VAL_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

print("✅ Dataset split completed")
print(f"Train rows: {len(train_df)}")
print(f"Validation rows: {len(val_df)}")
print(f"Test rows: {len(test_df)}")
