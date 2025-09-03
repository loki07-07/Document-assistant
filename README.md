# 📄 Smart Document Assistant

A **Streamlit-based NLP pipeline** for analyzing **Government Scheme PDFs** (short policy briefs, pamphlets, or exam prep notes).  

---

## 🚀 Features

- **📑 PDF Ingestion**: Upload up to 5 scheme-related PDFs.
- **📝 Preprocessing & Pattern Extraction**
  - Cleans and tokenizes text
  - Lemmatization + stopword removal
  - Extracts:
    - Monetary amounts (₹, Rs, lakh/crore, etc.)
    - Years
    - Scheme names (e.g., *PM Kisan Yojana*, *Swachh Bharat Mission*)
    - Percentages
- **🔧 Query Error Handling**
  - Spell checking and correction using **TextBlob**
- **🔤 Language Structure Analysis**
  - POS tagging
  - Named Entity Recognition (spaCy)
  - Visual POS distribution chart
- **🔍 Semantic Similarity**
  - TF–IDF + cosine similarity
  - Ranking of documents relevant to a query
  - Bar graph of similarity scores
- **🎯 Document Retrieval**
  - Query-based search across documents
  - Previews with extracted key information
  - Stats dashboard (document counts, lengths, processed words)

---

## 🛠️ Tech Stack

- **Python 3.8+**
- [Streamlit](https://streamlit.io/) – Web app framework  
- [NLTK](https://www.nltk.org/) – Tokenization, POS tagging, stopwords  
- [spaCy](https://spacy.io/) – Named Entity Recognition (NER)  
- [TextBlob](https://textblob.readthedocs.io/en/dev/) – Spell correction  
- [scikit-learn](https://scikit-learn.org/stable/) – TF–IDF, cosine similarity  
- [PyPDF2](https://pypi.org/project/PyPDF2/) – PDF text extraction  
- [Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/) – Charts & visualization  
- [WordCloud](https://amueller.github.io/word_cloud/) – Word cloud generation (optional)  

---

## 📂 Project Structure

NLP_EXAM/
│
├── data/ # (optional) store sample scheme PDFs
├── app.py # main Streamlit application (your script)
├── requirements.txt # Python dependencies
├── README.md # project documentation
└── .gitignore # ignore cache/venv/large files

yaml
Copy code

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/loki07-07/Document-assistant.git
   cd Document-assistant
Create virtual environment (recommended)

bash
Copy code
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
Install dependencies

bash
Copy code
pip install -r requirements.txt
Download required NLP models

bash
Copy code
python -m nltk.downloader punkt stopwords averaged_perceptron_tagger wordnet omw-1.4
python -m spacy download en_core_web_sm
▶️ Usage
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Open in browser (usually):

arduino
Copy code
http://localhost:8501
Upload up to 5 government scheme PDFs and explore the following tabs:

Preprocessing & Patterns

Query Error Handling

Language Structure

Semantic Similarity

Document Retrieval

## Demo

![Smart Document Assistant Demo](./output.gif)

GitHub: loki07-07

This project is for academic purposes (CHRIST University NLP Lab Exam).




---






Ask ChatGPT
