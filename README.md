# Smart Document Assistant - NLP Pipeline

A comprehensive Natural Language Processing system for analyzing government scheme documents, developed for academic demonstration purposes.

## Overview

This project implements a complete NLP pipeline that processes PDF documents and provides intelligent text analysis capabilities including pattern extraction, spell correction, language structure analysis, semantic similarity, and document retrieval.

## Features

### Core NLP Components
- **Text Preprocessing & Pattern Extraction** - Clean text data and extract monetary amounts, years, scheme names, and percentages
- **Query Error Handling** - Spell correction using TextBlob for improved user queries
- **Language Structure Analysis** - Part-of-speech tagging and Named Entity Recognition
- **Semantic Similarity Analysis** - TF-IDF based document similarity scoring
- **Document Retrieval System** - Intelligent document ranking and retrieval

### Visualization Capabilities
- **POS Tag Distribution Charts** - Visual representation of grammatical structures
- **Named Entity Charts** - Analysis of identified entities in documents
- **Word Clouds** - Visual word frequency representations
- **Similarity Bar Charts** - Document relevance scoring visualizations
- **Pattern Analysis Charts** - Years distribution and monetary amount analysis

## Installation

### Prerequisites
```bash
Python 3.7+
pip (Python package installer)
```

### Setup
1. Clone the repository:
```bash
git clone https://github.com/loki07-07/Document-assistant.git
cd Document-assistant
```



3. Download additional NLP models:
```bash
python -m spacy download en_core_web_sm
```

## Dependencies

```txt
nltk>=3.8.1
textblob>=0.17.1
scikit-learn>=1.3.0
spacy>=3.6.1
matplotlib>=3.7.2
seaborn>=0.12.2
PyPDF2>=3.0.1
pandas>=2.0.3
numpy>=1.24.3
wordcloud>=1.9.2
```

## Usage

### CLI Mode
Run the main application:
```bash
python nlp.py
```

### Available Options
1. **View patterns in a document** - Extract and display monetary amounts, years, scheme names
2. **Spell check a query** - Correct spelling errors in user input
3. **Analyze language structure** - POS tagging and NER analysis
4. **Check semantic similarity** - Compare query similarity with documents
5. **Retrieve relevant documents** - Find most relevant documents for a query
6. **Generate ALL charts for a document** - Create comprehensive visualizations
7. **Generate ALL charts for ALL documents** - Batch visualization generation
8. **View available chart files** - List generated visualization files

### Sample Workflow
1. Prepare a folder containing PDF documents of government schemes
2. Run the application and provide the folder path
3. Use menu options to analyze documents and generate visualizations
4. Check the `output/` folder for saved charts and analysis results

## File Structure

```
Document-assistant/
├── nlp.py                    # Main application file
├── README.md                 # This documentation
├── requirements.txt          # Python dependencies
├── data/                     # Sample PDF documents (optional)
├── output/                   # Generated visualization files
│   ├── pos_tags_*.png       # POS distribution charts
│   ├── entities_*.png       # Named entity charts
│   ├── wordcloud_*.png      # Word cloud visualizations
│   ├── similarity_*.png     # Similarity analysis charts
│   ├── years_*.png          # Years distribution charts
│   └── money_*.png          # Monetary pattern charts
└── .gitignore               # Git ignore rules
```

## Technical Implementation

### NLP Techniques Used
- **Text Preprocessing**: Tokenization, stopword removal, lemmatization
- **Pattern Matching**: Regular expressions for extracting structured information
- **Spell Correction**: TextBlob-based automatic correction
- **POS Tagging**: NLTK-based grammatical analysis
- **Named Entity Recognition**: spaCy-based entity extraction
- **TF-IDF Vectorization**: Document similarity computation
- **Cosine Similarity**: Semantic similarity measurement

### Visualization Libraries
- **Matplotlib**: Core plotting functionality
- **Seaborn**: Statistical visualizations
- **WordCloud**: Text visualization
- **Pandas**: Data manipulation for charts

## Academic Context

This project was developed for:
- **Institution**: CHRIST (Deemed to be University), Bangalore
- **Course**: MDS472C - Natural Language Processing
- **Programme**: MSc Data Science
- **Assessment**: ESE2 Component B - Lab Exam

## Sample Use Cases

### Government Document Analysis
- Process policy documents and scheme descriptions
- Extract key financial information and timelines
- Identify important entities and organizations
- Compare document similarity for policy analysis

### Text Mining Applications
- Automated information extraction from large document collections
- Query-based document retrieval systems
- Language pattern analysis for content categorization
- Spell-corrected search functionality

## Limitations and Considerations

- PDF text extraction quality depends on source document formatting
- spaCy NER model performance varies with domain-specific terminology
- TF-IDF similarity may not capture semantic relationships in specialized domains
- Spell correction works best with common English vocabulary



---

**Note**: This implementation demonstrates fundamental NLP concepts and is intended for educational evaluation. Production use would require additional validation, error handling, and security considerations.
