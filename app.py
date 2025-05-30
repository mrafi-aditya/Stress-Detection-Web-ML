import streamlit as st
import pandas as pd
import subprocess

st.set_page_config(page_title="Deteksi Stres dari Tweet", layout="centered")

st.markdown("<h1 style='text-align: center;'>🧠 Deteksi Stres dari Tweet</h1>", unsafe_allow_html=True)

# --- Styling ---
st.markdown("""
<style>
    .stTextInput > div > div > input {
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        font-size: 16px;
        border-radius: 8px;
        padding: 0.6em 1.5em;
        margin-top: 10px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.04);
    }
</style>
""", unsafe_allow_html=True)

# --- Input Username ---
username = st.text_input("Masukkan Username Twitter (tanpa @)", max_chars=50)

if st.button("🔍 Analisis Tweet"):
    if not username.strip():
        st.warning("⚠️ Silakan masukkan username terlebih dahulu.")
    else:
        username = username.strip().replace("@", "")
        st.info(f"🔁 Menganalisis tweet dari **@{username}**...")

        # Simpan ke username.txt
        with open("username.txt", "w") as f:
            f.write(username)

        # Jalankan crawling
        with st.spinner("Mengambil tweet..."):
            subprocess.run([
                "jupyter", "nbconvert", "--to", "notebook", "--execute",
                "crawling.ipynb", "--output", "executed_crawling.ipynb"
            ])

        # Jalankan analisis
        with st.spinner("Menganalisis stres..."):
            subprocess.run([
                "jupyter", "nbconvert", "--to", "notebook", "--execute",
                "analysis.ipynb", "--output", "executed_analysis.ipynb"
            ])

        # --- Baca CSV Hasil ---
        try:
            df = pd.read_csv("tweet_stress_classification_results.csv")

            st.success("✅ Analisis selesai!")

            # Tampilkan tabel tweet dan hasilnya
            st.subheader("📋 Tabel Hasil Deteksi")
            st.dataframe(df[["text", "label", "similarity_score"]].rename(columns={
                "text": "Tweet",
                "label": "Hasil",
                "similarity_score": "Similarity"
            }))

            # Hitung dan tampilkan ringkasan
            total = len(df)
            stress_count = (df["label"] == "stres").sum()
            non_stress_count = total - stress_count
            stress_pct = (stress_count / total) * 100
            non_stress_pct = 100 - stress_pct

            if stress_pct >= 50:
                conclusion = "Stres"
            else:
                conclusion = "Tidak Stres"

            st.subheader("📊 Ringkasan Analisis")
            st.markdown(f"- Jumlah Tweet: **{total}**")
            st.markdown(f"- Tweet Stres: **{stress_count}** ({stress_pct:.2f}%)")
            st.markdown(f"- Tweet Tidak Stres: **{non_stress_count}** ({non_stress_pct:.2f}%)")
            st.markdown(f"### 🔎 Kesimpulan: **{conclusion}**")

        except Exception as e:
            st.error(f"Gagal membaca hasil analisis: {e}")
