import pandas as pd
import numpy as np
import spacy
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from tqdm import tqdm
import os
import sys

class TextPreprocessor:
    """Handles text preprocessing including cleaning, tokenization, and lemmatization."""
    
    def __init__(self):
        self.nlp = self._load_spacy_model()
        self.stop_words = spacy.lang.en.stop_words.STOP_WORDS
    
    def _load_spacy_model(self):
        """Load spaCy model with error handling for downloading the model."""
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model 'en_core_web_sm'...")
            from spacy.cli import download
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        return nlp
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        if not isinstance(text, str) or not text.strip():
            return ""
            
        try:
            # Convert to lowercase and remove special characters
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            
            # Tokenize and lemmatize
            doc = self.nlp(text)
            lemmas = [token.lemma_ for token in doc 
                     if not token.is_stop and 
                     not token.is_punct and 
                     not token.is_space and
                     len(token.text) > 2]
            
            return ' '.join(lemmas)
        except Exception as e:
            print(f"Error processing text: {e}")
            return ""

    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Apply text cleaning to a DataFrame column."""
        tqdm.pandas(desc="Preprocessing text")
        df['cleaned_text'] = df[text_column].progress_apply(self.clean_text)
        return df

class KeywordExtractor:
    """Extracts keywords using TF-IDF and spaCy."""
    
    def __init__(self, max_features: int = 100):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        
    def extract_keywords_tfidf(self, texts: List[str], n_keywords: int = 10) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF."""
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get top keywords across all documents
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        keywords = [(feature_names[i], tfidf_scores[i]) 
                   for i in np.argsort(tfidf_scores)[-n_keywords:]]
        
        return sorted(keywords, key=lambda x: x[1], reverse=True)

class ThemeClusterer:
    """Clusters documents into themes using K-means."""
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
    
    def cluster_documents(self, texts: List[str]) -> Tuple[Dict[int, List[str]], np.ndarray]:
        """Cluster documents and return cluster terms and labels."""
        if not texts:
            return {}, np.array([])
            
        try:
            # Vectorize texts
            X = self.vectorizer.fit_transform(texts)
            
            # Fit K-means
            clusters = self.model.fit_predict(X)
            
            # Get top terms per cluster
            order_centroids = self.model.cluster_centers_.argsort()[:, ::-1]
            terms = self.vectorizer.get_feature_names_out()
            
            cluster_terms = {}
            for i in range(self.n_clusters):
                cluster_terms[i] = [terms[ind] for ind in order_centroids[i, :10]]
                
            return cluster_terms, clusters
            
        except Exception as e:
            print(f"Error in clustering: {e}")
            return {}, np.array([])

def process_reviews(file_path: str, n_clusters: int = 5) -> pd.DataFrame:
    """
    Process reviews with full NLP pipeline.
    
    Args:
        file_path: Path to the CSV file containing reviews
        n_clusters: Number of clusters for theme detection
        
    Returns:
        DataFrame with processed data and analysis results
    """
    try:
        # Load data
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        
        # Check required columns
        if 'Review Text' not in df.columns:
            raise ValueError("Input CSV must contain 'Review Text' column")
        
        # Initialize components
        print("Initializing NLP components...")
        preprocessor = TextPreprocessor()
        keyword_extractor = KeywordExtractor()
        clusterer = ThemeClusterer(n_clusters=n_clusters)
        
        # Preprocess text
        print("Preprocessing text...")
        df = preprocessor.preprocess_dataframe(df, 'Review Text')
        
        # Remove empty texts after cleaning
        df = df[df['cleaned_text'].str.strip().astype(bool)]
        
        if len(df) == 0:
            raise ValueError("No valid text data to analyze after cleaning")
        
        # Extract keywords
        print("\nExtracting keywords...")
        texts = df['cleaned_text'].tolist()
        keywords = keyword_extractor.extract_keywords_tfidf(texts)
        print("\nTop 10 Keywords:", [k[0] for k in keywords[:10]])
        
        # Cluster into themes
        print("\nClustering into themes...")
        cluster_terms, clusters = clusterer.cluster_documents(texts)
        
        if not cluster_terms:
            print("Warning: No themes could be identified")
            return df
            
        df['theme'] = clusters
        
        # Print theme terms
        print("\nTheme Terms:")
        for theme, terms in cluster_terms.items():
            print(f"Theme {theme + 1}:", ", ".join(terms[:5]) + "...")
        
        return df
        
    except Exception as e:
        print(f"Error in process_reviews: {str(e)}", file=sys.stderr)
        raise

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "data/Bank_of_Abyssinia_reviews.csv"
        
    output_file = "data/processed_reviews.csv"
    
    try:
        print(f"Starting processing of {input_file}")
        processed_df = process_reviews(input_file, n_clusters=5)
    
        if not processed_df.empty:
            processed_df.to_csv(output_file, index=False)
            print(f"\nProcessed data saved to {output_file}")
        else:
            print("No data was processed successfully.")
    
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
