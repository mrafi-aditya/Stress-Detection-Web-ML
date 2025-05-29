
import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- Fungsi Pembersih Tweet ---
def clean_tweet(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.lower().strip()

# --- Fungsi Mendapatkan Embedding IndoBERT ---
def get_indobert_embedding(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# --- Fungsi Klasifikasi Tweet ---
def classify_tweet(tweet, stress_phrases, non_stress_phrases, tokenizer, model, device):
    cleaned_tweet = clean_tweet(tweet)
    if not cleaned_tweet:
        return "tidak_stres", 0.0

    tweet_embedding = get_indobert_embedding(cleaned_tweet, tokenizer, model, device)

    stress_embeddings = [get_indobert_embedding(phrase, tokenizer, model, device) for phrase in stress_phrases]
    non_stress_embeddings = [get_indobert_embedding(phrase, tokenizer, model, device) for phrase in non_stress_phrases]

    stress_similarities = [cosine_similarity(tweet_embedding, emb)[0][0] for emb in stress_embeddings]
    non_stress_similarities = [cosine_similarity(tweet_embedding, emb)[0][0] for emb in non_stress_embeddings]

    max_stress_sim = max(stress_similarities) if stress_similarities else 0
    max_non_stress_sim = max(non_stress_similarities) if non_stress_similarities else 0

    if max_stress_sim > max_non_stress_sim:
        return "stres", max_stress_sim
    else:
        return "tidak_stres", max_non_stress_sim

# --- Inisialisasi Model IndoBERT ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1").to(device)

# --- Daftar Frasa Referensi ---
stress_phrases = [
    "saya merasa sedih", "saya stres", "saya cemas", "hidupku hancur", "aku ingin menyerah", "aku capek"
]
non_stress_phrases = [
    "saya bahagia", "saya senang", "aku merasa tenang", "aku bersyukur", "semuanya baik-baik saja"
]

# --- Fungsi Utama untuk Klasifikasi Dataset ---
def classify_tweets_from_csv(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"File tidak ditemukan: {input_path}")
        return

    df = pd.read_csv(input_path)
    if 'tweet' not in df.columns:
        print("Kolom 'tweet' tidak ditemukan pada file CSV.")
        return

    results = []
    for tweet in df['tweet']:
        label, score = classify_tweet(tweet, stress_phrases, non_stress_phrases, tokenizer, model, device)
        results.append({"tweet": tweet, "label": label, "score": score})

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    print(f"Hasil klasifikasi disimpan di: {output_path}")

# Contoh pemanggilan:
# classify_tweets_from_csv("tweets-data/sample.csv", "tweets-data/hasil_klasifikasi.csv")
