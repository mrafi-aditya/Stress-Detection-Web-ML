from flask import Flask, render_template, request
import subprocess
from stress_analysis import analisis_stres

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    username = request.form['username']
    
    # Simpan username ke file agar bisa dibaca oleh crawling.ipynb
    with open('username.txt', 'w') as f:
        f.write(username)
    
    # Jalankan crawling.ipynb
    subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "crawling.ipynb", "--output", "executed_crawling.ipynb"])

    # Lanjutkan ke analisis
    hasil = analisis_stres()
    return render_template('result.html', hasil=hasil)

if __name__ == '__main__':
    app.run(debug=True)
