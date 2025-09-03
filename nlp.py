import os
import re
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from collections import Counter
import PyPDF2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Optional: WordCloud (install with: pip install wordcloud)
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("⚠️ WordCloud not available. Install with: pip install wordcloud")

# ---- NLTK setup ----
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ---- spaCy ----
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    print("⚠️ spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None
    SPACY_AVAILABLE = False

# Create output directory
os.makedirs('output', exist_ok=True)

class SmartDocumentAssistant:
    def __init__(self):
        self.documents = {}
        self.preprocessed_docs = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF file"""
        try:
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    if page.extract_text():
                        text += page.extract_text()
            return text
        except Exception as e:
            print(f"Error reading PDF {file_path}: {str(e)}")
            return ""

    def preprocess_text(self, text):
        """Preprocess text"""
        text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower()
        tokens = word_tokenize(text)
        processed_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        return ' '.join(processed_tokens)

    def extract_patterns(self, text):
        """Extract patterns like scheme names, amounts, dates"""
        patterns = {}
        money_pattern = r'₹\s*[\d,]+|rs\.?\s*[\d,]+|rupees?\s+[\d,]+'
        patterns['monetary_amounts'] = re.findall(money_pattern, text, re.IGNORECASE)

        year_pattern = r'\b(?:19|20)\d{2}\b'
        patterns['years'] = re.findall(year_pattern, text)

        scheme_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*(?:Scheme|Yojana|Mission|Program|Initiative)\b'
        patterns['scheme_names'] = re.findall(scheme_pattern, text)

        percentage_pattern = r'\d+(?:\.\d+)?%'
        patterns['percentages'] = re.findall(percentage_pattern, text)

        return patterns

    def spell_check_and_correct(self, query):
        blob = TextBlob(query)
        corrected = str(blob.correct())
        return corrected, corrected != query

    def analyze_language_structure(self, text):
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        pos_counts = Counter([tag for word, tag in pos_tags])

        entities = []
        if SPACY_AVAILABLE:
            doc = nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]

        return pos_tags, pos_counts, entities

    def calculate_semantic_similarity(self, query):
        if not self.preprocessed_docs:
            return {}
        processed_query = self.preprocess_text(query)
        all_texts = [processed_query] + list(self.preprocessed_docs.values())
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
        sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        return dict(zip(self.preprocessed_docs.keys(), sims))

    def retrieve_relevant_documents(self, query, top_k=3):
        sims = self.calculate_semantic_similarity(query)
        return dict(sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_k])

    # =============== VISUALIZATION METHODS ===============

    def create_pos_chart(self, doc_name, pos_counts):
        """Create POS tag distribution chart"""
        plt.figure(figsize=(12, 6))
        top_pos = dict(pos_counts.most_common(15))
        
        plt.bar(top_pos.keys(), top_pos.values(), color='skyblue', edgecolor='navy', alpha=0.7)
        plt.title(f'POS Tag Distribution - {doc_name}', fontsize=16, fontweight='bold')
        plt.xlabel('POS Tags', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        filename = f"output/pos_tags_{doc_name.replace('.pdf', '')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 POS chart saved: {filename}")

    def create_entities_chart(self, doc_name, entities):
        """Create named entities chart"""
        if not entities:
            print(f"⚠️ No entities found in {doc_name}")
            return
            
        entity_types = Counter([ent[1] for ent in entities])
        
        plt.figure(figsize=(10, 6))
        top_entities = dict(entity_types.most_common(10))
        
        plt.bar(top_entities.keys(), top_entities.values(), color='lightcoral', edgecolor='darkred', alpha=0.7)
        plt.title(f'Named Entity Types - {doc_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Entity Types', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        filename = f"output/entities_{doc_name.replace('.pdf', '')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"🏷️ Entities chart saved: {filename}")

    def create_wordcloud(self, doc_name, text):
        """Create word cloud"""
        if not WORDCLOUD_AVAILABLE:
            print("⚠️ WordCloud not available")
            return
            
        preprocessed_text = self.preprocess_text(text)
        
        wordcloud = WordCloud(
            width=800, height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(preprocessed_text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud - {doc_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = f"output/wordcloud_{doc_name.replace('.pdf', '')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"☁️ Word cloud saved: {filename}")

    def create_similarity_chart(self, query, similarities):
        """Create similarity bar chart"""
        plt.figure(figsize=(10, 6))
        
        docs = list(similarities.keys())
        scores = list(similarities.values())
        
        colors = plt.cm.viridis([s for s in scores])
        bars = plt.barh(docs, scores, color=colors, edgecolor='black', alpha=0.8)
        
        plt.title(f'Document Similarity Scores\nQuery: "{query}"', fontsize=14, fontweight='bold')
        plt.xlabel('Similarity Score', fontsize=12)
        plt.ylabel('Documents', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        
        # Add score labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        clean_query = re.sub(r'[^\w\s]', '', query)[:20]
        filename = f"output/similarity_{clean_query.replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Similarity chart saved: {filename}")

    def create_years_chart(self, doc_name, years):
        """Create years histogram"""
        if not years:
            print(f"⚠️ No years found in {doc_name}")
            return
            
        years_int = [int(year) for year in years]
        
        plt.figure(figsize=(10, 6))
        plt.hist(years_int, bins=20, color='gold', edgecolor='orange', alpha=0.7)
        plt.title(f'Years Distribution - {doc_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Years', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        filename = f"output/years_{doc_name.replace('.pdf', '')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📅 Years chart saved: {filename}")

    def create_money_chart(self, doc_name, money_amounts):
        """Create money mentions chart"""
        if not money_amounts:
            print(f"⚠️ No monetary amounts found in {doc_name}")
            return
            
        # Count frequency of different money patterns
        money_types = Counter()
        for amount in money_amounts:
            if '₹' in amount:
                money_types['Rupee Symbol (₹)'] += 1
            elif 'rs' in amount.lower():
                money_types['Rs. Format'] += 1
            elif 'rupee' in amount.lower():
                money_types['Rupees Text'] += 1
        
        plt.figure(figsize=(8, 6))
        types = list(money_types.keys())
        counts = list(money_types.values())
        
        plt.pie(counts, labels=types, autopct='%1.1f%%', startangle=90, 
               colors=['lightgreen', 'lightblue', 'lightcoral'])
        plt.title(f'Money Format Distribution - {doc_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f"output/money_{doc_name.replace('.pdf', '')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💰 Money chart saved: {filename}")

    def generate_all_charts_for_document(self, doc_name):
        """Generate all possible charts for a document"""
        print(f"\n🎨 Generating charts for: {doc_name}")
        text = self.documents[doc_name]
        
        # 1. Pattern extraction charts
        patterns = self.extract_patterns(text)
        if patterns['years']:
            self.create_years_chart(doc_name, patterns['years'])
        if patterns['monetary_amounts']:
            self.create_money_chart(doc_name, patterns['monetary_amounts'])
        
        # 2. Language structure charts
        pos_tags, pos_counts, entities = self.analyze_language_structure(text[:2000])  # First 2000 chars
        self.create_pos_chart(doc_name, pos_counts)
        if entities:
            self.create_entities_chart(doc_name, entities)
        
        # 3. Word cloud
        if WORDCLOUD_AVAILABLE:
            self.create_wordcloud(doc_name, text)


# ------------------- CLI Interface -------------------

def main():
    assistant = SmartDocumentAssistant()

    print("\n📄 Smart Document Assistant")
    print("=========================================================\n")

    # Step 1: Upload PDFs
    folder = input("Enter the folder path containing your PDF documents: ").strip()
    if not os.path.isdir(folder):
        print("❌ Invalid folder path.")
        return

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder, file)
            print(f"Processing {file}...")
            text = assistant.extract_text_from_pdf(file_path)
            if text:
                assistant.documents[file] = text
                assistant.preprocessed_docs[file] = assistant.preprocess_text(text)

    if not assistant.documents:
        print("❌ No PDF documents loaded. Exiting.")
        return

    print(f"\n✅ Loaded {len(assistant.documents)} documents.\n")

    # Step 2: Menu
    while True:
        print("\nChoose an option:")
        print("1️⃣  View patterns in a document")
        print("2️⃣  Spell check a query")
        print("3️⃣  Analyze language structure")
        print("4️⃣  Check semantic similarity")
        print("5️⃣  Retrieve relevant documents")
        print("6️⃣  Generate ALL charts for a document")
        print("7️⃣  Generate ALL charts for ALL documents")
        print("8️⃣  View available chart files")
        print("0️⃣  Exit")

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            doc_name = input(f"Enter document name from {list(assistant.documents.keys())}: ")
            if doc_name in assistant.documents:
                patterns = assistant.extract_patterns(assistant.documents[doc_name])
                print("\n🔍 Extracted Patterns:")
                for key, vals in patterns.items():
                    print(f"- {key}: {vals[:5]}")
                    
                # Ask if user wants charts for patterns
                if input("\nGenerate pattern charts? (y/n): ").lower() == 'y':
                    if patterns['years']:
                        assistant.create_years_chart(doc_name, patterns['years'])
                    if patterns['monetary_amounts']:
                        assistant.create_money_chart(doc_name, patterns['monetary_amounts'])
            else:
                print("❌ Document not found.")

        elif choice == "2":
            query = input("Enter your query: ")
            corrected, changed = assistant.spell_check_and_correct(query)
            print(f"Corrected Query: {corrected}")
            if changed:
                print("✅ Corrections applied.")
            else:
                print("✅ No corrections needed.")

        elif choice == "3":
            doc_name = input(f"Enter document name from {list(assistant.documents.keys())}: ")
            if doc_name in assistant.documents:
                pos_tags, pos_counts, entities = assistant.analyze_language_structure(
                    assistant.documents[doc_name][:2000]
                )
                print("\nTop POS Tags:", pos_counts.most_common(10))
                print("Named Entities:", entities[:10])
                
                # Generate structure charts
                if input("\nGenerate structure charts? (y/n): ").lower() == 'y':
                    assistant.create_pos_chart(doc_name, pos_counts)
                    if entities:
                        assistant.create_entities_chart(doc_name, entities)
            else:
                print("❌ Document not found.")

        elif choice == "4":
            query = input("Enter query for similarity analysis: ")
            sims = assistant.calculate_semantic_similarity(query)
            print("\n📈 Similarity Scores:")
            for doc, score in sorted(sims.items(), key=lambda x: x[1], reverse=True):
                print(f"{doc}: {score:.4f}")
                
            # Generate similarity chart
            if input("\nGenerate similarity chart? (y/n): ").lower() == 'y':
                assistant.create_similarity_chart(query, sims)

        elif choice == "5":
            query = input("Enter query for document retrieval: ")
            top_k = int(input("How many top documents? "))
            results = assistant.retrieve_relevant_documents(query, top_k)
            print("\n🏆 Retrieved Documents:")
            for doc, score in results.items():
                print(f"{doc} (score: {score:.4f})")

        elif choice == "6":
            doc_name = input(f"Enter document name from {list(assistant.documents.keys())}: ")
            if doc_name in assistant.documents:
                assistant.generate_all_charts_for_document(doc_name)
            else:
                print("❌ Document not found.")

        elif choice == "7":
            print("\n🎨 Generating ALL charts for ALL documents...")
            for doc_name in assistant.documents.keys():
                assistant.generate_all_charts_for_document(doc_name)
            print("\n✅ All charts generated!")

        elif choice == "8":
            print("\n📁 Available chart files in output/ folder:")
            if os.path.exists('output'):
                files = [f for f in os.listdir('output') if f.endswith('.png')]
                if files:
                    for i, file in enumerate(files, 1):
                        print(f"{i:2d}. {file}")
                else:
                    print("No chart files found.")
            else:
                print("Output folder doesn't exist yet.")

        elif choice == "0":
            print("👋 Exiting. Goodbye!")
            print(f"📂 Check the 'output/' folder for saved visualizations!")
            break
        else:
            print("❌ Invalid choice. Try again.")


if __name__ == "__main__":
    main()
