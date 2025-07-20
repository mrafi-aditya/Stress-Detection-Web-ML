import streamlit as st
import pandas as pd
import os
import subprocess
import time
import papermill as pm  # 🔄 Tambahkan papermill

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

        # Simpan username
        with open("username.txt", "w") as f:
            f.write(username)

        # Hapus file lama (jika ada)
        hasil_path = "hasil/hasil_analisis.csv"
        if os.path.exists(hasil_path):
            os.remove(hasil_path)

        # Jalankan crawling.ipynb pakai papermill
        with st.spinner("🐦 Mengambil tweet... (bisa makan waktu 1-2 menit)"):
            try:
                pm.execute_notebook(
                    input_path="crawling.ipynb",
                    output_path="executed_crawling.ipynb",
                    parameters={}
                )
            except Exception as e:
                st.error(f"❌ Gagal menjalankan notebook crawling.ipynb: {e}")
                st.stop()

        # Jalankan analisis
        with st.spinner("🔬 Menganalisis stres..."):
            try:
                result = subprocess.run(
                    ["python", "classify_tweets.py"],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    st.error("❌ Gagal saat eksekusi classify_tweets.py")
                    st.text(result.stderr)
                    st.stop()
            except Exception as e:
                st.error(f"Gagal menjalankan classify_tweets.py: {e}")
                st.stop()

        # Baca hasil
        if not os.path.exists(hasil_path):
            st.error("❌ Hasil analisis tidak ditemukan.")
            st.stop()

        try:
            df = pd.read_csv(hasil_path)
            st.success("✅ Analisis selesai!")

            st.subheader("📋 Tabel Hasil Deteksi")
            st.dataframe(df[["clean_text", "label", "similarity_score"]].rename(columns={
                "clean_text": "Tweet",
                "label": "Hasil",
                "similarity_score": "Similarity"
            }))

            # Ringkasan
            total = len(df)
            stress_count = (df["label"] == "stres").sum()
            non_stress_count = total - stress_count
            stress_pct = (stress_count / total) * 100
            non_stress_pct = 100 - stress_pct

            conclusion = "Stres" if stress_pct >= 50 else "Tidak Stres"

            st.subheader("📊 Ringkasan Analisis")
            st.markdown(f"- Jumlah Tweet: **{total}**")
            st.markdown(f"- Tweet Stres: **{stress_count}** ({stress_pct:.2f}%)")
            st.markdown(f"- Tweet Tidak Stres: **{non_stress_count}** ({non_stress_pct:.2f}%)")
            st.markdown(f"### 🔎 Kesimpulan: **{conclusion}**")

        except Exception as e:
            st.error(f"Gagal membaca hasil analisis: {e}")
