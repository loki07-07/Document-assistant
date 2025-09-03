import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import PyPDF2
import io
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

# Load spaCy model (fallback to basic processing if not available)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

class SmartDocumentAssistant:
    def __init__(self):
        self.documents = {}
        self.preprocessed_docs = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.tfidf_matrix = None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def extract_text_from_pdf(self, uploaded_file):
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def preprocess_text(self, text):
        """Comprehensive text preprocessing"""
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(processed_tokens)
    
    def extract_patterns(self, text):
        """Extract patterns like scheme names, amounts, dates"""
        patterns = {}
        
        # Extract monetary amounts
        money_pattern = r'₹\s*[\d,]+|rs\.?\s*[\d,]+|rupees?\s+[\d,]+'
        patterns['monetary_amounts'] = re.findall(money_pattern, text, re.IGNORECASE)
        
        # Extract years
        year_pattern = r'\b(19|20)\d{2}\b'
        patterns['years'] = re.findall(year_pattern, text)
        
        # Extract scheme-like names (capitalized words)
        scheme_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*(?:Scheme|Yojana|Mission|Program|Initiative)\b'
        patterns['scheme_names'] = re.findall(scheme_pattern, text)
        
        # Extract percentages
        percentage_pattern = r'\d+(?:\.\d+)?%'
        patterns['percentages'] = re.findall(percentage_pattern, text)
        
        return patterns
    
    def spell_check_and_correct(self, query):
        """Handle spelling errors in user queries using generic spell correction"""
        blob = TextBlob(query)
        corrected = str(blob.correct())
        was_corrected = corrected != query
        
        return corrected, was_corrected
    
    def analyze_language_structure(self, text):
        """Analyze POS tags and named entities"""
        # POS Tagging
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        
        # Count POS tags
        pos_counts = Counter([tag for word, tag in pos_tags])
        
        # Named Entity Recognition using spaCy if available
        entities = []
        if nlp:
            doc = nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        return pos_tags, pos_counts, entities
    
    def calculate_semantic_similarity(self, query, documents):
        """Calculate semantic similarity between query and documents"""
        if not documents:
            return {}
        
        # Preprocess query
        processed_query = self.preprocess_text(query)
        
        # Combine query with documents for TF-IDF
        all_texts = [processed_query] + list(documents.values())
        
        # Calculate TF-IDF
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
        
        # Calculate cosine similarity between query and each document
        query_vector = tfidf_matrix[0:1]
        doc_vectors = tfidf_matrix[1:]
        
        similarities = cosine_similarity(query_vector, doc_vectors).flatten()
        
        # Create similarity scores dictionary
        similarity_scores = {}
        doc_names = list(documents.keys())
        for i, score in enumerate(similarities):
            similarity_scores[doc_names[i]] = score
        
        return similarity_scores
    
    def retrieve_relevant_documents(self, query, top_k=3):
        """Retrieve most relevant documents based on query"""
        if not self.preprocessed_docs:
            return {}
        
        # Calculate similarities
        similarities = self.calculate_semantic_similarity(query, self.preprocessed_docs)
        
        # Sort by similarity score
        sorted_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Return top k documents
        return dict(sorted_docs[:top_k])

def main():
    st.set_page_config(
        page_title="Smart Document Assistant",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("🤖 Smart Document Assistant - NLP Pipeline")
    st.markdown("*A comprehensive NLP system for government scheme document analysis*")
    
    # Initialize session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = SmartDocumentAssistant()
    
    assistant = st.session_state.assistant
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📁 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload Government Scheme PDFs", 
            type=['pdf'], 
            accept_multiple_files=True,
            help="Upload up to 5 PDF documents of Indian government schemes"
        )
        
        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} documents")
            
            # Process uploaded files
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in assistant.documents:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        text = assistant.extract_text_from_pdf(uploaded_file)
                        if text:
                            assistant.documents[uploaded_file.name] = text
                            assistant.preprocessed_docs[uploaded_file.name] = assistant.preprocess_text(text)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1️⃣ Preprocessing & Patterns", 
        "2️⃣ Query Error Handling", 
        "3️⃣ Language Structure", 
        "4️⃣ Semantic Similarity", 
        "5️⃣ Document Retrieval"
    ])
    
    # Tab 1: Preprocessing & Pattern Extraction
    with tab1:
        st.header("📝 Preprocessing & Pattern Extraction")
        
        if assistant.documents:
            selected_doc = st.selectbox("Select document for analysis:", list(assistant.documents.keys()))
            
            if selected_doc:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Text (First 500 chars)")
                    st.text_area("", assistant.documents[selected_doc][:500], height=200, key="original_text")
                
                with col2:
                    st.subheader("Preprocessed Text (First 500 chars)")
                    st.text_area("", assistant.preprocessed_docs[selected_doc][:500], height=200, key="processed_text")
                
                # Pattern extraction
                st.subheader("🔍 Extracted Patterns")
                patterns = assistant.extract_patterns(assistant.documents[selected_doc])
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Monetary Amounts", len(patterns['monetary_amounts']))
                    if patterns['monetary_amounts']:
                        st.write(patterns['monetary_amounts'][:3])
                
                with col2:
                    st.metric("Years Found", len(patterns['years']))
                    if patterns['years']:
                        st.write(list(set(patterns['years']))[:3])
                
                with col3:
                    st.metric("Scheme Names", len(patterns['scheme_names']))
                    if patterns['scheme_names']:
                        st.write(patterns['scheme_names'][:3])
                
                with col4:
                    st.metric("Percentages", len(patterns['percentages']))
                    if patterns['percentages']:
                        st.write(patterns['percentages'][:3])
        else:
            st.info("Please upload PDF documents to begin analysis.")
    
    # Tab 2: Query Error Handling
    with tab2:
        st.header("🔧 Query Error Handling")
        
        st.subheader("Test Spell Correction")
        user_query = st.text_input(
            "Enter query (try with spelling mistakes):", 
            placeholder="e.g., 'jan dhan yojna beneficieries'"
        )
        
        if user_query:
            corrected_query, was_corrected = assistant.spell_check_and_correct(user_query)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Query:**")
                st.write(user_query)
            
            with col2:
                st.write("**Corrected Query:**")
                if was_corrected:
                    st.success(corrected_query)
                    st.info("✅ Spelling corrections applied")
                else:
                    st.write(corrected_query)
                    st.info("✅ No corrections needed")
    
    # Tab 3: Language Structure Analysis
    with tab3:
        st.header("🔤 Language Structure Analysis")
        
        if assistant.documents:
            selected_doc = st.selectbox("Select document for structure analysis:", list(assistant.documents.keys()), key="struct_doc")
            
            if selected_doc:
                # Analyze a sample of the document
                sample_text = assistant.documents[selected_doc][:1000]  # First 1000 characters
                pos_tags, pos_counts, entities = assistant.analyze_language_structure(sample_text)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 POS Tag Distribution")
                    if pos_counts:
                        # Create bar chart for top POS tags
                        top_pos = dict(pos_counts.most_common(10))
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.bar(top_pos.keys(), top_pos.values())
                        ax.set_xlabel("POS Tags")
                        ax.set_ylabel("Frequency")
                        ax.set_title("Top 10 POS Tags")
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                
                with col2:
                    st.subheader("🏷️ Named Entities")
                    if entities:
                        entity_df = pd.DataFrame(entities, columns=['Entity', 'Label'])
                        st.dataframe(entity_df.head(10))
                    else:
                        st.info("Install spaCy model for NER: python -m spacy download en_core_web_sm")
                
                # Sample POS tags
                st.subheader("🔖 Sample POS Tags")
                sample_pos = pos_tags[:20]  # First 20 words
                pos_df = pd.DataFrame(sample_pos, columns=['Word', 'POS Tag'])
                st.dataframe(pos_df)
        else:
            st.info("Please upload documents first.")
    
    # Tab 4: Semantic Similarity
    with tab4:
        st.header("🔍 Semantic Similarity Analysis")
        
        if assistant.preprocessed_docs:
            query_for_similarity = st.text_input(
                "Enter query for similarity analysis:", 
                placeholder="e.g., 'financial inclusion banking services'"
            )
            
            if query_for_similarity:
                similarities = assistant.calculate_semantic_similarity(query_for_similarity, assistant.preprocessed_docs)
                
                if similarities:
                    # Create DataFrame for better visualization
                    sim_df = pd.DataFrame(
                        list(similarities.items()), 
                        columns=['Document', 'Similarity Score']
                    ).sort_values('Similarity Score', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📈 Similarity Scores")
                        st.dataframe(sim_df, use_container_width=True)
                    
                    with col2:
                        st.subheader("📊 Similarity Visualization")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.barh(sim_df['Document'], sim_df['Similarity Score'])
                        ax.set_xlabel('Similarity Score')
                        ax.set_title('Document Similarity Scores')
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Show most similar document content preview
                    most_similar = sim_df.iloc[0]['Document']
                    st.subheader(f"📄 Preview of Most Similar Document: {most_similar}")
                    st.text_area("Content Preview", assistant.documents[most_similar][:500], height=150)
        else:
            st.info("Please upload and process documents first.")
    
    # Tab 5: Document Retrieval
    with tab5:
        st.header("🎯 Document Retrieval System")
        
        if assistant.preprocessed_docs:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                search_query = st.text_input(
                    "Enter search query:", 
                    placeholder="e.g., 'farmer welfare scheme rural development'"
                )
            
            with col2:
                top_k = st.slider("Number of documents to retrieve:", 1, len(assistant.documents), 3)
            
            if search_query:
                # Apply spell correction
                corrected_query, _ = assistant.spell_check_and_correct(search_query)
                
                if corrected_query != search_query:
                    st.info(f"Query corrected to: '{corrected_query}'")
                
                # Retrieve relevant documents
                relevant_docs = assistant.retrieve_relevant_documents(corrected_query, top_k)
                
                if relevant_docs:
                    st.subheader("🏆 Retrieved Documents")
                    
                    for i, (doc_name, score) in enumerate(relevant_docs.items(), 1):
                        with st.expander(f"#{i} {doc_name} (Score: {score:.4f})"):
                            st.write("**Content Preview:**")
                            st.write(assistant.documents[doc_name][:800] + "...")
                            
                            # Extract and show patterns from this document
                            patterns = assistant.extract_patterns(assistant.documents[doc_name])
                            if any(patterns.values()):
                                st.write("**Key Information:**")
                                if patterns['scheme_names']:
                                    st.write(f"• Schemes: {', '.join(patterns['scheme_names'][:3])}")
                                if patterns['monetary_amounts']:
                                    st.write(f"• Amounts: {', '.join(patterns['monetary_amounts'][:3])}")
                                if patterns['years']:
                                    st.write(f"• Years: {', '.join(set(patterns['years']))}")
                else:
                    st.warning("No relevant documents found for the query.")
            
            # Search Statistics
            if assistant.documents:
                st.subheader("📊 Document Collection Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Documents", len(assistant.documents))
                
                with col2:
                    total_chars = sum(len(doc) for doc in assistant.documents.values())
                    st.metric("Total Characters", f"{total_chars:,}")
                
                with col3:
                    avg_doc_length = total_chars / len(assistant.documents) if assistant.documents else 0
                    st.metric("Avg Doc Length", f"{avg_doc_length:.0f}")
                
                with col4:
                    total_words = sum(len(doc.split()) for doc in assistant.preprocessed_docs.values())
                    st.metric("Total Processed Words", f"{total_words:,}")
        else:
            st.info("Please upload documents to use the retrieval system.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        **Smart Document Assistant** - Built for CHRIST University MDS472C NLP Lab Exam
        
        **Features Demonstrated:**
        - ✅ Text Preprocessing & Pattern Extraction
        - ✅ Spell Correction & Query Error Handling  
        - ✅ Language Structure Analysis (POS, NER)
        - ✅ Semantic Similarity using TF-IDF
        - ✅ Document Retrieval System
        """
    )

if __name__ == "__main__":
    main()