import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

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

# --- Calculate Similarity Score ---
def calculate_similarity_score(df_tweets, tokenizer, model_name="indobenchmark/indobert-base-p1"):
    """
    Calculate similarity score between tweets using sentence embeddings
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model for embeddings
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    # Get embeddings for all tweets
    embeddings = []
    
    with torch.no_grad():
        for text in df_tweets['clean_text']:
            # Tokenize
            inputs = tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get embeddings
            outputs = model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(embedding[0])
    
    embeddings = np.array(embeddings)
    
    # Calculate pairwise cosine similarity
    similarity_matrix = cosine_similarity(embeddings)
    
    # Calculate average similarity for each tweet (excluding self-similarity)
    similarity_scores = []
    for i in range(len(similarity_matrix)):
        # Get similarities with other tweets (excluding self)
        similarities = np.concatenate([similarity_matrix[i][:i], similarity_matrix[i][i+1:]])
        avg_similarity = np.mean(similarities) if len(similarities) > 0 else 0.0
        similarity_scores.append(avg_similarity)
    
    return similarity_scores

# --- Inference Function ---
def classify_tweets(tweet_file, model_path, model_name="indobenchmark/indobert-base-p1", batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load label encoder
    label_encoder = LabelEncoder()
    label_encoder_classes_path = os.path.join(os.path.dirname(model_path), 'label_encoder_classes.npy')
    if not os.path.exists(label_encoder_classes_path):
        raise FileNotFoundError(f"Label encoder classes file not found at {label_encoder_classes_path}")
    label_encoder.classes_ = np.load(label_encoder_classes_path, allow_pickle=True)

    # Load model
    model = IndoBERTClassifier(model_name, num_labels=len(label_encoder.classes_)).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load and preprocess tweets
    df_tweets = pd.read_csv(tweet_file)
    df_tweets['clean_text'] = df_tweets['full_text'].apply(clean_tweet)
    texts = df_tweets['clean_text'].values

    # Create dataset and dataloader
    dataset = TweetDataset(texts, [0]*len(texts), tokenizer)  # Dummy labels for inference
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Perform inference
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)

    # Decode predictions
    predicted_labels = label_encoder.inverse_transform(predictions)
    df_tweets['label'] = predicted_labels

    # Calculate similarity scores
    print("Menghitung similarity score...")
    similarity_scores = calculate_similarity_score(df_tweets, tokenizer, model_name)
    df_tweets['similarity_score'] = similarity_scores

    # Calculate statistics
    total_tweets = len(df_tweets)
    stress_count = sum(df_tweets['label'] == 'stres')
    stress_percentage = (stress_count / total_tweets) * 100
    final_conclusion = "Stres" if stress_count > (total_tweets - stress_count) else "Tidak Stres"
    
    # Calculate average similarity score
    avg_similarity = np.mean(similarity_scores)

    # Print results
    print(f"\nHasil Analisis Stres dari {total_tweets} Tweet:")
    print(f"Jumlah Tweet Stres: {stress_count} ({stress_percentage:.2f}%)")
    print(f"Jumlah Tweet Tidak Stres: {total_tweets - stress_count} ({100 - stress_percentage:.2f}%)")
    print(f"Average Similarity Score: {avg_similarity:.4f}")
    print(f"Kesimpulan Akhir: {final_conclusion}")

    # Print sample classifications
    print("\nContoh Hasil Klasifikasi (10 Pertama):")
    for i, row in df_tweets.head(10).iterrows():
        print(f"Tweet: {row['clean_text']}")
        print(f"Label: {row['label']}")
        print(f"Similarity Score: {row['similarity_score']:.4f}")
        print("-" * 50)

    # Save results to CSV
    output_dir = "hasil"
    os.makedirs(output_dir, exist_ok=True)  # Create hasil directory if it doesn't exist
    output_path = os.path.join(output_dir, "hasil_analisis.csv")
    df_tweets.to_csv(output_path, index=False)
    print(f"\nHasil analisis disimpan ke: {output_path}")

    return df_tweets

# --- Main Execution ---
if __name__ == "__main__":
    tweet_file = "tweets-data/save_tweets.csv"
    model_path = "indobert_stress_classifier.pt"
    classify_tweets(tweet_file, model_path)