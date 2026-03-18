# Research Paper Categorizer

A 3-class NLP pipeline that categorizes research papers into:

- `APPLIED`
- `THEORETICAL`
- `SURVEY`

The current trained model in this repository is a **TF-IDF + Multinomial Logistic Regression** classifier built from paper title + abstract text.

## What We Used

### Data Sources

Source datasets in `Dataset/`:

- `applied_papers_1000.csv` (1000 papers)
- `theory_papers_1000.csv` (1000 papers)
- `survey_papers_901.csv` (901 papers)

Initial total: **2901 papers**.

### Tech Stack

- Python 3.11 (conda environment used during development)
- Data processing: `pandas`, `numpy`, `scipy`
- Classical ML: `scikit-learn`
- Model/artifact persistence: `joblib`, `json`
- Deep embedding experiments: `torch`, `transformers`, `sentence-transformers`
- Visualization and analysis: `matplotlib`
- Notebook workflow for collection, EDA, and training

Dependencies are listed in `requirements.txt`.

## What We Developed

### 1) Dataset Collection Workflow

Notebook sequence in `Dataset collection notebooks/`:

- `01_collect_theory_arxiv_500.ipynb`
- `02_collect_survey_arxiv.ipynb`
- `03_collect_applied_arxiv.ipynb`
- `04_collect_applied_arxiv_1000.ipynb`

These notebooks build class-specific datasets from arXiv and save CSV/JSON outputs.

### 2) Data Preparation + EDA

Notebook: `05_data_prep_eda.ipynb`

Key steps implemented:

- Built `model_text = title + " " + abstract`
- Audited/removed conflicting and duplicate records
- Generated class distribution and token-level EDA tables
- Created train/validation/test splits

Dedup/audit summary from `data/dedup_audit_summary.csv`:

- Rows after missing-text drop: **2901**
- Conflicting IDs: **2**
- Rows removed due to conflicting labels: **4**
- Rows removed due to duplicate IDs: **42**
- Rows removed due to duplicate text fingerprints: **0**
- Final rows after dedup: **2855**

Final label counts from `data/eda_label_counts.csv`:

- `APPLIED`: 998
- `THEORETICAL`: 998
- `SURVEY`: 859

Split files in `data/splits/`:

- `train_papers.csv`: 1998 rows (+ header)
- `val_papers.csv`: 428 rows (+ header)
- `test_papers.csv`: 429 rows (+ header)

### 3) TF-IDF Model Training

Notebook: `06_tfidf_model.ipynb`

Implemented model pipeline:

- `TfidfVectorizer` with:
  - `ngram_range=(1, 2)`
  - `min_df=3`
  - `max_df=0.95`
  - `sublinear_tf=True`
  - `strip_accents="unicode"`
- `LogisticRegression` (multiclass softmax behavior with `lbfgs`)
- Validation search over `C` in `[0.25, 0.5, 1.0, 2.0, 4.0]`
- Selected best `C = 4.0`

Saved artifacts in `models/tfidf/`:

- `tfidf_vectorizer.joblib`
- `tfidf_logreg.joblib`
- `label_encoder.joblib`
- `tfidf_metrics.json`
- `tfidf_val_results.csv`
- `tfidf_classification_report.csv`

Also saved:

- `plots/models/tfidf_confusion_matrix.png`

### 4) SPECTER Embedding Experiment

Notebook: `07_specter_model.ipynb`

- Includes an embedding-based alternative pipeline (`SPECTER` + Logistic Regression)
- Uses cached embeddings in `data/embeddings/` when available
- In this repository snapshot, only TF-IDF trained artifacts are present under `models/`

## Current Model Performance (TF-IDF)

From `models/tfidf/tfidf_metrics.json` and `models/tfidf/tfidf_classification_report.csv`:

- Accuracy: **0.9814**
- Macro F1: **0.9815**
- Weighted F1: **0.9814**

Per-class F1:

- `APPLIED`: 0.9735
- `SURVEY`: 0.9845
- `THEORETICAL`: 0.9866

## How To Use The Trained Model

### 1) Install Dependencies

```bash
pip install -r requirements.txt
```

If you use conda, activate your environment first:

```bash
conda activate finetuning
```

### 2) Predict From CLI

Use the included script `predict_tfidf_category.py`:

```bash
python predict_tfidf_category.py \
  --title "Attention Is All You Need" \
  --abstract "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks ..." \
  --top-k 3
```

Sample output format:

```text
Predicted category: APPLIED
Top probabilities:
  - APPLIED: 0.4689
  - SURVEY: 0.3693
  - THEORETICAL: 0.1618
```

### 3) Interactive Mode

Run without `--title`/`--abstract` and provide inputs when prompted:

```bash
python predict_tfidf_category.py
```

### 4) Use A Custom Model Directory

```bash
python predict_tfidf_category.py \
  --model-dir models/tfidf \
  --title "Your title" \
  --abstract "Your abstract"
```

### 5) Programmatic Usage (Python)

```python
from pathlib import Path
from predict_tfidf_category import TfidfPaperCategorizer

predictor = TfidfPaperCategorizer(model_dir=Path("models/tfidf"))
result = predictor.predict(
    title="A Survey of Foundation Models",
    abstract="This paper surveys recent progress in foundation models...",
    top_k=3,
)

print(result)
```

## Reproducible Notebook Order

1. Dataset collection notebooks (`Dataset collection notebooks/`)
2. `05_data_prep_eda.ipynb`
3. `06_tfidf_model.ipynb`
4. Optional: `07_specter_model.ipynb`

## Repository Layout

```text
.
├── Dataset/
├── Dataset collection notebooks/
├── data/
│   ├── combined_papers_cleaned.csv
│   ├── dedup_*.csv
│   ├── eda_*.csv
│   ├── splits/
│   └── embeddings/
├── models/
│   └── tfidf/
├── plots/
│   └── models/
├── 05_data_prep_eda.ipynb
├── 06_tfidf_model.ipynb
├── 07_specter_model.ipynb
├── predict_tfidf_category.py
└── requirements.txt
```

## Notes

- Inference expects the same text construction used in training: `title + " " + abstract`.
- If `python` is not found in your shell, run with your environment's full executable path.
- The production-ready artifacts currently available in this repository are the TF-IDF model artifacts in `models/tfidf/`.