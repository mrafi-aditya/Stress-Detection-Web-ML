import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm

# --- Dataset Kustom untuk Klasifikasi Tweet ---
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- Model Klasifikasi IndoBERT ---
class IndoBERTClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# --- Fungsi Bersihkan Tweet ---
def clean_tweet(text):
    import re
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.lower().strip()
    return text

# --- Latih Model ---
def train_model(dataset_path, output_dir, model_name="indobenchmark/indobert-base-p1", epochs=3, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Jika tokenizer belum punya pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token

    # Buat direktori output jika belum ada
    os.makedirs(output_dir, exist_ok=True)

    # --- Muat dan praproses dataset
    print("Membaca dan membersihkan dataset...")
    df = pd.read_csv(dataset_path)
    df['text'] = df['text'].apply(clean_tweet)
    texts = df['text'].values
    labels = df['label'].values

    # --- Encode label
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    np.save(os.path.join(output_dir, 'label_encoder_classes.npy'), label_encoder.classes_)
    print("Label classes disimpan.")

    # --- Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )

    # --- Dataset dan Dataloader
    train_dataset = TweetDataset(train_texts, train_labels, tokenizer)
    val_dataset = TweetDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # --- Model
    model = IndoBERTClassifier(model_name, num_labels=len(label_encoder.classes_)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    # --- Loop training
    print("Mulai pelatihan...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", unit="batch")
        for batch in train_progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Rata-rata Training Loss = {avg_train_loss:.4f}")

        # --- Validasi
        model.eval()
        total_val_loss = 0
        correct_preds = 0
        total_preds = 0

        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation", unit="batch")
        with torch.no_grad():
            for batch in val_progress:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)
                total_val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                correct_preds += (preds == labels).sum().item()
                total_preds += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_preds / total_preds * 100
        print(f"Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f}, Val Accuracy = {val_accuracy:.2f}%")

    # --- Simpan model
    model_save_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Model disimpan ke {model_save_path}")

    return tokenizer, label_encoder

# --- Eksekusi Utama ---
if __name__ == "__main__":
    dataset_path = "dataset/real_dataset_stress.csv"
    output_dir = "trained_model"
    train_model(dataset_path, output_dir)
