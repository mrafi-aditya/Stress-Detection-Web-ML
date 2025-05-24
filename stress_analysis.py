import pandas as pd
import os
import re
import joblib

def bersihkan_teks(teks):
    teks = teks.lower()
    teks = re.sub(r'http\S+|www\S+', '', teks)
    teks = re.sub(r'[^a-z\s]', '', teks)
    return teks.strip()

def analisis_stres(tweet_path='tweets-data/stress_tweets.csv', model_path='model/stress_model.pkl'):
    if not os.path.exists(tweet_path):
        return "File tweet tidak ditemukan."
    if not os.path.exists(model_path):
        return "Model tidak ditemukan."

    # Load pipeline langsung (tanpa akses 'model' atau 'vectorizer')
    pipeline = joblib.load(model_path)

    # Load tweet CSV
    tweet_df = pd.read_csv(tweet_path)
    if 'text' not in tweet_df.columns:
        tweet_df = pd.read_csv(tweet_path, names=['text', 'kategori', 'sumber'], header=0)

    tweet_df['clean'] = tweet_df['text'].astype(str).apply(bersihkan_teks)

    # Prediksi langsung
    predictions = pipeline.predict(tweet_df['clean'])
    probs = pipeline.predict_proba(tweet_df['clean'])[:, 1] if hasattr(pipeline, "predict_proba") else [None] * len(predictions)

    hasil = list(zip(tweet_df['text'], predictions, probs))
    return hasil
