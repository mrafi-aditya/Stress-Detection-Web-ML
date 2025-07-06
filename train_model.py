import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm  # Import tqdm for progress bar

# --- Custom Dataset for Tweet Classification ---
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
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- IndoBERT Classifier Model ---
class IndoBERTClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super(IndoBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# --- Clean Tweet Function ---
def clean_tweet(text):
    import re
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = text.lower().strip()  # Convert to lowercase and strip
    return text

# --- Train the Model ---
def train_model(dataset_path, model_save_path, model_name="indobenchmark/indobert-base-p1", epochs=3, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load and preprocess dataset
    df = pd.read_csv(dataset_path)
    df['text'] = df['text'].apply(clean_tweet)
    texts = df['text'].values
    labels = df['kategori'].values

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Save label encoder classes
    np.save(os.path.join(os.path.dirname(model_save_path), 'label_encoder_classes.npy'), label_encoder.classes_)

    # Split dataset
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, encoded_labels, test_size=0.2, random_state=42
    )

    # Create datasets
    train_dataset = TweetDataset(train_texts, train_labels, tokenizer)
    val_dataset = TweetDataset(val_texts, val_labels, tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    model = IndoBERTClassifier(model_name, num_labels=len(label_encoder.classes_)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop with tqdm
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        # Add tqdm progress bar for training batches
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Update progress bar with current loss
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    return tokenizer, label_encoder

# --- Main Execution ---
if __name__ == "__main__":
    dataset_path = "dataset/real_dataset_stress.csv"
    model_save_path = "indobert_stress_classifier.pt"
    train_model(dataset_path, model_save_path)