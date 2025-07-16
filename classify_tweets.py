import os
import re
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

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
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- Model Klasifikasi IndoBERT ---
class IndoBERTClassifier(nn.Module):
    def __init__(self, base_model_name, num_labels):
        super(IndoBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(base_model_name)
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
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.lower().strip()

# --- Hitung Skor Kemiripan ---
def calculate_similarity_score(df_tweets, tokenizer, base_model_name="indobenchmark/indobert-base-p1"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(base_model_name).to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for text in df_tweets['clean_text']:
            inputs = tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(embedding[0])

    embeddings = np.array(embeddings)
    similarity_matrix = cosine_similarity(embeddings)

    similarity_scores = []
    for i in range(len(similarity_matrix)):
        similarities = np.concatenate([similarity_matrix[i][:i], similarity_matrix[i][i+1:]])
        avg_similarity = np.mean(similarities) if len(similarities) > 0 else 0.0
        similarity_scores.append(avg_similarity)
    return similarity_scores

# --- Fungsi Inferensi ---
def classify_tweets(tweet_file, hf_repo="aditonomy/stres-detection-indobert", batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load tokenizer dari repo HF
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

    # --- Download label encoder classes.npy
    print("Mengunduh label encoder dari Hugging Face Hub...")
    label_classes_path = hf_hub_download(repo_id=hf_repo, filename="label_encoder_classes.npy")
    label_classes = np.load(label_classes_path, allow_pickle=True)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = label_classes

    # --- Download model weights
    print("Mengunduh model dari Hugging Face Hub...")
    model_weights_path = hf_hub_download(repo_id=hf_repo, filename="pytorch_model.bin")

    # --- Load model
    model = IndoBERTClassifier(
        base_model_name="indobenchmark/indobert-base-p1",
        num_labels=len(label_encoder.classes_)
    ).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    # --- Load dan bersihkan data
    df_tweets = pd.read_csv(tweet_file)
    df_tweets['clean_text'] = df_tweets['full_text'].apply(clean_tweet)
    texts = df_tweets['clean_text'].values

    # --- Buat dataset & dataloader
    dataset = TweetDataset(texts, [0]*len(texts), tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # --- Prediksi label
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)

    predicted_labels = label_encoder.inverse_transform(predictions)
    df_tweets['label'] = predicted_labels

    # --- Hitung skor kemiripan
    print("Menghitung skor kemiripan embedding...")
    similarity_scores = calculate_similarity_score(df_tweets, tokenizer, "indobenchmark/indobert-base-p1")
    df_tweets['similarity_score'] = similarity_scores

    # --- Statistik hasil
    total_tweets = len(df_tweets)
    stress_count = sum(df_tweets['label'] == 'stres')
    stress_percentage = (stress_count / total_tweets) * 100
    final_conclusion = "Stres" if stress_count > (total_tweets - stress_count) else "Tidak Stres"
    avg_similarity = np.mean(similarity_scores)

    # --- Cetak hasil ringkas
    print(f"\nHasil Analisis Stres dari {total_tweets} Tweet:")
    print(f"Jumlah Tweet Stres: {stress_count} ({stress_percentage:.2f}%)")
    print(f"Jumlah Tweet Tidak Stres: {total_tweets - stress_count} ({100 - stress_percentage:.2f}%)")
    print(f"Rata-rata Skor Kemiripan: {avg_similarity:.4f}")
    print(f"Kesimpulan Akhir: {final_conclusion}")

    # --- Contoh output
    print("\nContoh Hasil Klasifikasi (10 Pertama):")
    for i, row in df_tweets.head(10).iterrows():
        print(f"Tweet: {row['clean_text']}")
        print(f"Label: {row['label']}")
        print(f"Skor Kemiripan: {row['similarity_score']:.4f}")
        print("-" * 50)

    # --- Simpan hasil ke CSV
    os.makedirs("hasil", exist_ok=True)
    output_path = os.path.join("hasil", "hasil_analisis.csv")
    df_tweets.to_csv(output_path, index=False)
    print(f"\nHasil analisis disimpan ke: {output_path}")

    return df_tweets

# --- Eksekusi Utama ---
if __name__ == "__main__":
    tweet_file = "tweets-data/save_tweets.csv"
    classify_tweets(tweet_file, hf_repo="aditonomy/stres-detection-indobert")
