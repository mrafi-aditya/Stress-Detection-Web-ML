# Stress Indication Detection with IndoBERT

A Streamlit application for exploring stress indications in Indonesian-language text from CSV files. Developed as an Informatics scientific writing project, it connects text preprocessing, an IndoBERT classifier, and downloadable analysis results.

## Analysis workflow

1. Upload a CSV containing a `full_text` column.
2. Remove links, mentions, hashtags, digits, and punctuation, then lowercase the text.
3. Tokenize the cleaned text and run a PyTorch classifier built on `indobenchmark/indobert-base-p1`.
4. Calculate a mean cosine-similarity score between text embeddings.
5. Display labels, similarity scores, and a label-count summary; export the result as CSV.

**Stack:** Python · pandas · NumPy · PyTorch · Transformers · scikit-learn · Hugging Face Hub · Streamlit

## Run locally

```sh
git clone https://github.com/mrafi-aditya/Stress-Detection-Web-ML.git
cd Stress-Detection-Web-ML
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on macOS/Linux, then:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the address printed by Streamlit, normally `http://localhost:8501`. Inference uses CUDA when available and otherwise runs on the CPU. Dependencies are currently unpinned, so a fresh installation may require compatibility adjustments.

## Model dependencies

The application downloads these assets at analysis time:

| Source | Required assets |
|---|---|
| [indobenchmark/indobert-base-p1](https://huggingface.co/indobenchmark/indobert-base-p1) | Base model and tokenizer |
| [aditonomy/stres-detection-indobert](https://huggingface.co/aditonomy/stres-detection-indobert) | `pytorch_model.bin` and `label_encoder_classes.npy` |

An internet connection and access to both model repositories are required on the first run. Only load model weights and serialized label files from sources you trust. No social-media API credential is required for the CSV analysis path in `app.py`.

## Example input

Save this synthetic example as a UTF-8 CSV:

```csv
full_text
"Saya merasa kewalahan dengan tugas yang menumpuk."
"Hari ini saya menikmati waktu bersama teman."
```

Use at least two non-empty text rows: the current similarity calculation compares each row with other rows. The returned CSV adds `clean_text`, `label`, and `similarity_score` to the input columns.

## Project map

| Path | Purpose |
|---|---|
| [app.py](app.py) | UI, preprocessing, model loading, classification, and export |
| [requirements.txt](requirements.txt) | Runtime dependencies |
| `dataset/` | Project data files |
| `hasil/` | Previously generated analysis output |
| `tweets-data/` | Previously collected text files |

The UI also links to a separate collection notebook; data collection is outside this repository's inference workflow. Supply only text you have permission to process and share.

## Interpretation and limitations

Labels indicate patterns learned by the model, not a clinical diagnosis or a determination of a person's mental health. The summary uses the proportion of predicted `stres` labels with a 50% cutoff. Similarity measures resemblance between the submitted texts, not classification confidence or severity.

Embedding comparisons produce a pairwise matrix, so time and memory requirements grow with the batch size. Large CSVs, empty input, and single-row input need additional validation before broader use. Training, benchmark reproduction, and production deployment are not provided by this repository.
