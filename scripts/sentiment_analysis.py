import pandas as pd
import numpy as np
from transformers import pipeline
from tqdm import tqdm
from typing import Dict, Tuple, List
import os

# Initialize the sentiment analysis pipeline
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess the reviews data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Convert rating to integer if it's stored as string
    if 'Rating' in df.columns:
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').fillna(0).astype(int)
    
    # Ensure we have the required columns
    required_columns = ['Review Text', 'Rating']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the data")
    
    return df

def analyze_sentiment_batch(texts: List[str], batch_size: int = 32) -> List[Dict]:
    """Analyze sentiment for a batch of texts using the pre-trained model."""
    sentiment_analyzer = pipeline("sentiment-analysis", 
                                model=SENTIMENT_MODEL,
                                device=-1)  # Use CPU (-1), change to 0 for GPU if available
    
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiment"):
        batch = texts[i:i + batch_size]
        batch_results = sentiment_analyzer(batch)
        results.extend(batch_results)
    return results

def analyze_sentiments(file_path: str) -> pd.DataFrame:
    """
    Perform sentiment analysis on reviews and aggregate by bank and rating.
    
    Args:
        file_path (str): Path to the CSV file containing reviews
    
    Returns:
        pd.DataFrame: DataFrame with aggregated sentiment scores
    """
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(file_path)
    
    print("\nAnalyzing sentiment for reviews...")
    # Get sentiment scores
    texts = df['Review Text'].astype(str).tolist()
    sentiment_results = analyze_sentiment_batch(texts)
    
    # Add sentiment scores to DataFrame
    df['sentiment_score'] = [result['score'] * (1 if result['label'] == 'POSITIVE' else -1) 
                           for result in sentiment_results]
    df['sentiment_label'] = [result['label'] for result in sentiment_results]
    
    # Calculate sentiment statistics by bank and rating
    print("\nAggregating results...")
    if 'Bank/App Name' not in df.columns:
        df['bank_name'] = os.path.basename(file_path).split('_reviews_')[0]
    
    cleaned_dir = "data/cleaned_reviews"
    os.makedirs(cleaned_dir, exist_ok=True)
    
    cleaned_file_name = os.path.basename(file_path).replace(".csv", "_cleaned.csv")
    cleaned_path = os.path.join(cleaned_dir, cleaned_file_name)
    df.to_csv(cleaned_path, index=False)
    print(f"Cleaned data saved to: {cleaned_path}")

    # Group by bank and rating
    aggregation = {
        'sentiment_score': ['count', 'mean', 'std', 'min', 'max'],
        'sentiment_label': lambda x: (x == 'POSITIVE').mean() * 100  # % positive
    }
    
    results = df.groupby(['Bank/App Name', 'Rating']).agg(aggregation)
    results.columns = ['review_count', 'mean_sentiment', 'std_sentiment', 
                      'min_sentiment', 'max_sentiment', 'percent_positive']
    
    # Reset index for better readability
    results = results.reset_index()
    
    return results

def main():
    # Example usage
    input_files = [
                    "data/Bank_of_Abyssinia_reviews.csv", 
                    "data/Commercial_Bank_of_Ethiopia_reviews.csv", 
                    "data/Dashen_Bank_reviews.csv"
                  ]
    
    for input_file in input_files:
        try:
            print(f"\nProcessing file: {input_file}")
            analyze_sentiments(input_file)
        except Exception as e:
            print(f"\nError processing {input_file}: {str(e)}")
    
if __name__ == "__main__":
    main()
