# Research Paper Categorizer

Classifies research papers into three categories:
- APPLIED
- THEORETICAL
- SURVEY

The project includes:

- A Streamlit UI for PDF upload and classification
- PDF title and abstract extraction using Groq
- Training and analysis notebooks for TF-IDF based classification

## Project Structure

- app.py: Streamlit UI entrypoint
- src/extractor.py: PDF extraction logic (uses Groq API)
- models/tfidf_scratch/scratch_logreg_artifacts.joblib: Saved model artifacts used by UI
- data_cleaning.ipynb: Data cleaning workflow
- research_paper_classification.ipynb: Model development notebook
- hyperparameter_tuning_pipeline.ipynb: Hyperparameter experiments
- dataset_wordcloud_evaluation.ipynb: Dataset wordcloud and EDA visuals

## Installation (Python venv)

1. Clone the repository and move into the project folder.

2. Create a virtual environment:

```bash
python3 -m venv .venv
```

3. Activate the virtual environment:

Linux/macOS:
```bash
source .venv/bin/activate
```

Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

4. Upgrade pip and install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Environment Variables (.env)

Create a .env file in the project root with your Groq key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

This key is required for the UI because title and abstract extraction in src/extractor.py depends on Groq.

## Run the UI

With the virtual environment active:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501), upload a research paper PDF, and view the predicted category with confidence scores.

## Requirements Check

requirements.txt has been aligned with the current project usage, including:

- Core ML and data libraries (numpy, pandas, scikit-learn, scipy, joblib)
- UI and extraction libraries (streamlit, groq, python-dotenv, PyMuPDF)
- Notebook/EDA libraries (matplotlib, seaborn, wordcloud)
- Dataset collection support (requests)

## Notes

- Keep .env in the project root so python-dotenv can load GROQ_API_KEY automatically.
- If GROQ_API_KEY is missing, extraction will fail and UI classification from uploaded PDFs will not proceed.