import streamlit as st
import tempfile
import numpy as np
import joblib
from pathlib import Path

from src.extractor import FirstPageTitleAbstractExtractor

# Page config
st.set_page_config(page_title="Research Paper Classifier", layout="wide")
st.title("📄 Research Paper Classifier")
st.markdown("Upload a research paper PDF to classify it into one of three categories: **Applied**, **Survey**, or **Theory**.")

# Load model artifacts
@st.cache_resource
def load_model():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    artifact_path = script_dir / 'models' / 'tfidf_scratch' / 'scratch_logreg_artifacts.joblib'
    if not artifact_path.exists():
        st.error(f"Model artifacts not found at {artifact_path}")
        return None
    return joblib.load(artifact_path)

@st.cache_resource
def load_extractor():
    return FirstPageTitleAbstractExtractor()

# Load preprocessing function
def preprocess_text(text):
    """Match the preprocessing from the notebook."""
    import re
    from nltk.stem import PorterStemmer
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    
    stop_words = set(ENGLISH_STOP_WORDS)
    stemmer = PorterStemmer()
    
    def tokens(t):
        return re.findall(r'[a-z]+', str(t).lower())
    
    tok = tokens(text)
    no_stop = [w for w in tok if w not in stop_words]
    stemmed = [stemmer.stem(w) for w in no_stop]
    return ' '.join(stemmed)

def softmax(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

# Load artifacts
artifact = load_model()
extractor = load_extractor()

if artifact is None:
    st.stop()

vectorizer = artifact['vectorizer']
W = artifact['weights']
b = artifact['bias']
i2lab = artifact['index_to_label']
label_names = artifact['label_names']

# UI
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    try:
        with st.spinner("📖 Extracting title and abstract..."):
            result = extractor.extract(tmp_path)
            title = result.title
            abstract = result.abstract
            source = result.source
        
        st.success(f"✓ Extraction successful (source: {source})")
        
        # Display extracted content
        with st.expander("📋 Extracted Content", expanded=True):
            st.markdown("**Title:**")
            st.write(title)
            st.markdown("**Abstract:**")
            st.write(abstract)
        
        # Classify
        with st.spinner("🤖 Classifying paper..."):
            combined_text = (title + ' ' + abstract).strip()
            processed_text = preprocess_text(combined_text)
            X = vectorizer.transform([processed_text]).toarray()
            probs = softmax(X @ W + b)[0]
            pred_idx = int(np.argmax(probs))
            pred_category = i2lab[pred_idx]
            confidence = probs[pred_idx]
        
        # Display results
        st.success("✓ Classification complete")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Category", pred_category)
        with col2:
            st.metric("Confidence", f"{confidence:.2%}")
        
        # Show all probabilities
        st.subheader("Classification Probabilities")
        probs_df = __import__('pandas').DataFrame({
            'Category': [i2lab[i] for i in range(len(label_names))],
            'Probability': probs
        }).sort_values('Probability', ascending=False).reset_index(drop=True)
        
        # Create a bar chart
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['#2ecc71' if i == pred_idx else '#3498db' for i in range(len(label_names))]
        ax.barh(probs_df['Category'], probs_df['Probability'], color=colors)
        ax.set_xlabel('Probability')
        ax.set_title('Classification Probabilities')
        ax.set_xlim([0, 1])
        st.pyplot(fig)
        
        st.dataframe(probs_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)
else:
    st.info("👆 Upload a PDF file using the sidebar to get started")
