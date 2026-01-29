import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LENGTH

class SymptomDataset(Dataset):
    def __init__(self, csv_file, tokenizer):
        self.df = pd.read_csv(csv_file)
        self.texts = self.df["symptoms_text"].tolist()
        self.labels = self.df["disease_id"].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)
