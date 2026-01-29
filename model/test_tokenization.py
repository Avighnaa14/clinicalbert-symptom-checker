from dataset_loader import SymptomDataset, load_tokenizer

TRAIN_FILE = "data/processed/train.csv"

tokenizer = load_tokenizer()
dataset = SymptomDataset(TRAIN_FILE, tokenizer)

sample = dataset[0]

print("✅ Sample tokenization successful")
print("Input IDs shape:", sample["input_ids"].shape)
print("Attention mask shape:", sample["attention_mask"].shape)
print("Label:", sample["labels"])
