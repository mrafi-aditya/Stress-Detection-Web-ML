# app.py
import streamlit as st
import pandas as pd
import os
import re
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

# --- Styling dan Judul ---
st.set_page_config(page_title="Deteksi Stres dari Tweet", layout="centered")
st.markdown("<h1 style='text-align: center;'>🧠 Deteksi Stres dari Tweet (via CSV)</h1>", unsafe_allow_html=True)

# --- Dataset Kustom ---
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
            text, add_special_tokens=True, max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- Model IndoBERT ---
class IndoBERTClassifier(nn.Module):
    def __init__(self, base_model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# --- Cleaning Text ---
def clean_tweet(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+|[^\w\s]|\d+", "", text)
    return text.lower().strip()

# --- Similarity Score ---
def calculate_similarity_score(df_tweets, tokenizer, base_model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(base_model_name).to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for text in df_tweets['clean_text']:
            inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy()[0])

    sim_matrix = cosine_similarity(embeddings)
    return [np.mean(np.delete(row, i)) for i, row in enumerate(sim_matrix)]

# --- Klasifikasi ---
def classify_tweets_from_csv(uploaded_file, hf_repo="aditonomy/stres-detection-indobert", batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    classes_path = hf_hub_download(hf_repo, filename="label_encoder_classes.npy")
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(classes_path, allow_pickle=True)

    model_path = hf_hub_download(hf_repo, filename="pytorch_model.bin")
    model = IndoBERTClassifier("indobenchmark/indobert-base-p1", len(label_encoder.classes_)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    df = pd.read_csv(uploaded_file)
    if "full_text" not in df.columns:
        st.error("❌ Kolom 'full_text' tidak ditemukan dalam file CSV.")
        return None

    df["clean_text"] = df["full_text"].apply(clean_tweet)
    texts = df["clean_text"].values
    dataset = TweetDataset(texts, [0]*len(texts), tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_ids"].to(device)
            masks = batch["attention_mask"].to(device)
            outputs = model(inputs, masks)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)

    df["label"] = label_encoder.inverse_transform(predictions)
    df["similarity_score"] = calculate_similarity_score(df, tokenizer, "indobenchmark/indobert-base-p1")
    return df

st.markdown("""
    ℹ️ **Belum punya file CSV tweet?**

    Kamu bisa gunakan alat bantu crawling tweet di Google Colab berikut:

    👉 [Buka Colab: Crawling Tweet](https://colab.research.google.com/drive/13k_4n4dTa68h7PKUY2lt-eZicy81zCLt?copy)

    Setelah selesai crawling, kamu akan mendapatkan file `.csv` yang bisa diupload ke sini.
    """)

# --- Upload dan Analisis ---
uploaded_file = st.file_uploader("📂 Upload file CSV yang berisi kolom `full_text`", type=["csv"])

if uploaded_file and st.button("🔍 Analisis Tweet"):
    with st.spinner("⏳ Sedang menganalisis..."):
        df_result = classify_tweets_from_csv(uploaded_file)
        if df_result is not None:
            st.success("✅ Analisis selesai!")

            st.subheader("📋 Tabel Hasil Deteksi")
            st.dataframe(df_result[["clean_text", "label", "similarity_score"]].rename(columns={
                "clean_text": "Tweet", "label": "Hasil", "similarity_score": "Similarity"
            }))

            total = len(df_result)
            stress_count = (df_result["label"] == "stres").sum()
            stress_pct = (stress_count / total) * 100
            conclusion = "Stres" if stress_pct >= 50 else "Tidak Stres"

            st.subheader("📊 Ringkasan Analisis")
            st.markdown(f"- Jumlah Tweet: **{total}**")
            st.markdown(f"- Tweet Stres: **{stress_count}** ({stress_pct:.2f}%)")
            st.markdown(f"- Tweet Tidak Stres: **{total - stress_count}** ({100 - stress_pct:.2f}%)")
            st.markdown(f"### 🔎 Kesimpulan: **{conclusion}**")

            st.download_button(
                label="💾 Unduh Hasil Analisis",
                data=df_result.to_csv(index=False).encode("utf-8"),
                file_name="hasil_analisis.csv",
                mime="text/csv"
            )
