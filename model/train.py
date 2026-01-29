import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os

from dataset_loader import SymptomDataset, load_tokenizer
from load_model import load_tokenizer_and_model
from config import BATCH_SIZE, LEARNING_RATE, EPOCHS


TRAIN_FILE = "data/processed/train.csv"
VAL_FILE = "data/processed/val.csv"
SAVE_DIR = "saved_model"

os.makedirs(SAVE_DIR, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load tokenizer & model
tokenizer = load_tokenizer()
model = load_tokenizer_and_model()[1]
model.to(device)

# Load datasets
train_dataset = SymptomDataset(TRAIN_FILE, tokenizer)
val_dataset = SymptomDataset(VAL_FILE, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Optimizer & scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# Training loop
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Average training loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total
    print(f"Validation accuracy: {val_accuracy:.4f}")

# Save trained model
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print("\n✅ Training complete")
print(f"Model saved to → {SAVE_DIR}")
