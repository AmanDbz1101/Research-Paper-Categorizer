# 📄 Research Paper Categorizer

An end-to-end Natural Language Processing (NLP) system that automatically classifies academic research papers into one of three categories:

* **APPLIED** – Papers proposing or evaluating methods in practical domains.
* **THEORETICAL** – Papers focused on formal results, proofs, and foundational analysis.
* **SURVEY** – Papers that review and summarize existing research.

The project combines classical NLP techniques with machine learning to provide fast and interpretable research paper categorization.

---

## ✨ Features

* Upload research papers directly through a **Streamlit web interface**
* Extract paper **title and abstract from PDFs** using the Groq API
* Automatic classification into:

  * Applied
  * Theoretical
  * Survey
* Display prediction confidence scores
* Complete data collection, cleaning, training, and evaluation pipeline
* Reproducible notebooks and saved model artifacts

---

## 📊 Model Performance

| Metric           | Value                     |
| ---------------- | ------------------------- |
| Accuracy         | **98.14%**                |
| Macro F1 Score   | **0.98**                  |
| Training Dataset | **2,855 labelled papers** |
| Test Set Size    | **429 papers**            |

The results demonstrate that classical machine learning methods such as TF-IDF and Logistic Regression remain highly effective for domain-specific document classification tasks.

---

## 🏗️ Project Pipeline

```text
arXiv Data Collection
        ↓
Data Cleaning & Preparation
        ↓
Text Preprocessing
        ↓
TF-IDF Vectorization
        ↓
Multinomial Logistic Regression
        ↓
Model Evaluation
        ↓
Saved Artifacts & Deployment
        ↓
Streamlit Application
```

---

## 🧠 Methodology

### 1. Data Collection

Research papers were collected from the **arXiv API** using category-specific collection notebooks.

The dataset contains papers from multiple Computer Science domains, including:

* Machine Learning
* Computer Vision
* Logic in Computer Science
* Computational Complexity
* Formal Languages
* Survey Papers collected from curated repositories

---

### 2. Data Cleaning

The collected datasets undergo:

* Schema harmonization
* Duplicate removal
* Conflicting label resolution
* Missing value handling
* Text normalization

The final cleaned dataset is exported as:

```text
data/combined_papers_cleaned.csv
```

---

### 3. Text Preprocessing

The text preprocessing pipeline includes:

* Lowercasing
* Tokenization
* Stopword removal
* Porter Stemming
* Reconstruction into cleaned text

---

### 4. Feature Engineering

The project uses **TF-IDF Vectorization** with:

* Unigrams and Bigrams
* Maximum Features: 12,000
* Minimum Document Frequency: 3
* Sublinear TF Scaling
* Unicode Accent Stripping

---

### 5. Classification Model

Two implementations were developed:

1. **Multinomial Logistic Regression from Scratch**
2. **Scikit-Learn Logistic Regression** for hyperparameter tuning and deployment.

---

## 📁 Project Structure

```text
Research Paper Categorizer/
├── app.py
├── extractor.py
├── README.md
├── requirements.txt
├── data/
│   ├── combined_papers_cleaned.csv
├── Dataset/
│   ├── applied_papers_1000.csv
│   ├── applied_papers_1000.json
│   ├── survey_papers_901.csv
│   ├── survey_papers_901.json
│   ├── theory_papers_1000.csv
│   └── theory_papers_1000.json
├── Dataset collection notebooks/
│   ├── 01_collect_theory_arxiv.ipynb
│   ├── 02_collect_survey_arxiv.ipynb
│   ├── 03_collect_applied_arxiv_1000.ipynb

├── models/
│   └── tfidf_scratch/
│       └── scratch_logreg_artifacts.joblib
├── data_cleaning.ipynb
├── dataset_wordcloud_evaluation.ipynb
├── hyperparameter_tuning_pipeline.ipynb
└── research_paper_classification.ipynb
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AmanDbz1101/research-paper-categorizer.git
cd research-paper-categorizer
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

### 3. Activate the Environment

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

The Groq API is used to extract the paper's title and abstract from uploaded PDF files before classification.

---

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the URL shown in your terminal (typically `http://localhost:8501`) and upload a research paper PDF.

The application will:

1. Extract the title and abstract.
2. Preprocess the text.
3. Generate TF-IDF features.
4. Predict the paper category.
5. Display confidence scores.

---

## 🎯 Use Cases

* Academic research assistants
* Literature review automation
* Digital libraries
* Research management systems
* Educational tools for categorizing academic papers

---

## 🔮 Future Improvements

* Support for additional research categories
* Deep learning and Transformer-based models
* Explainable predictions using SHAP/LIME
* Batch classification of multiple PDFs
* Full-text paper classification instead of title and abstract only
* Integration with citation and recommendation systems

---


