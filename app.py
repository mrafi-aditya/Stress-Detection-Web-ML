from flask import Flask, render_template, request
import subprocess
import json  # untuk baca hasil dari file

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    username = request.form['username']

    # Simpan username ke file
    with open('username.txt', 'w') as f:
        f.write(username)

    # Jalankan crawling.ipynb
    subprocess.run([
        "jupyter", "nbconvert", "--to", "notebook", "--execute",
        "crawling.ipynb", "--output", "executed_crawling.ipynb"
    ])

    # Jalankan analysis.ipynb
    subprocess.run([
        "jupyter", "nbconvert", "--to", "notebook", "--execute",
        "analysis.ipynb", "--output", "executed_analysis.ipynb"
    ])

    # Baca hasil dari file JSON (misalnya hasil disimpan sebagai hasil.json oleh analysis.ipynb)
    with open('hasil.json', 'r', encoding='utf-8') as f:
        hasil = json.load(f)  # list of [tweet, label, prob]

    return render_template('result.html', hasil=hasil)

if __name__ == '__main__':
    app.run(debug=True)
