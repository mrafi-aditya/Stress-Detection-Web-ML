import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Fungsi pembersihan teks
def bersihkan_teks(teks):
    teks = teks.lower()
    teks = re.sub(r'http\S+|www\S+', '', teks)
    teks = re.sub(r'[^a-z\s]', '', teks)
    return teks.strip()

# Load dataset
df = pd.read_csv("dataset/dataset_stress.csv")

# Pastikan kolom bernama 'text' dan 'kategori'
df = df[['text', 'kategori']].dropna()

# Ubah label ke 0 (tidak stres) dan 1 (stres)
df['label'] = df['kategori'].map({'tidak_stres': 0, 'stres': 1})

# Bersihkan teks
df['clean'] = df['text'].apply(bersihkan_teks)

# Split
X = df['clean']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat pipeline: vectorizer + classifier
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Latih model
pipeline.fit(X_train, y_train)

# Simpan ke file
joblib.dump(pipeline, 'model/stress_model.pkl')
print("Model berhasil disimpan di model/stress_model.pkl")
