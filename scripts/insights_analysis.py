import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
from typing import List, Dict, Tuple
import os
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_theme()

def load_cleaned_data() -> pd.DataFrame:
    """Load and combine all cleaned review files."""
    cleaned_dir = "data/cleaned_reviews"
    all_data = []
    
    for file in os.listdir(cleaned_dir):
        if file.endswith("_cleaned.csv"):
            file_path = os.path.join(cleaned_dir, file)
            df = pd.read_csv(file_path)
            all_data.append(df)
    
    if not all_data:
        raise FileNotFoundError("No cleaned review files found!")
    
    return pd.concat(all_data, ignore_index=True)

def extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from text."""
    # Remove special characters and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Remove common stop words
    stop_words = {'the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'with', 'on', 'at', 'from', 'by', 'this', 'that', 'it', 'its', 'app', 'bank', 'banking'}
    words = text.split()
    return [word for word in words if word not in stop_words and len(word) > 2]

def analyze_drivers_and_pain_points(df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """Analyze key drivers and pain points from reviews."""
    # Combine all reviews
    all_reviews = ' '.join(df['Review Text'].astype(str))
    keywords = extract_keywords(all_reviews)
    
    # Get sentiment for each review
    df['textblob_sentiment'] = df['Review Text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    
    # Separate positive and negative reviews
    positive_reviews = df[df['textblob_sentiment'] > 0.2]['Review Text'].astype(str)
    negative_reviews = df[df['textblob_sentiment'] < -0.2]['Review Text'].astype(str)
    
    # Extract keywords for positive and negative reviews
    positive_keywords = extract_keywords(' '.join(positive_reviews))
    negative_keywords = extract_keywords(' '.join(negative_reviews))
    
    # Count keyword frequencies
    drivers = dict(Counter(positive_keywords).most_common(10))
    pain_points = dict(Counter(negative_keywords).most_common(10))
    
    return drivers, pain_points

def create_visualizations(df: pd.DataFrame, drivers: Dict, pain_points: Dict):
    """Create various visualizations for insights."""
    # Create output directory for plots
    os.makedirs("visualizations", exist_ok=True)
    
    # 1. Sentiment Distribution by Bank
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Bank/App Name', y='sentiment_score')
    plt.xticks(rotation=45)
    plt.title('Sentiment Score Distribution by Bank')
    plt.tight_layout()
    plt.savefig('visualizations/sentiment_by_bank.png')
    plt.close()
    
    # 2. Rating Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Rating', hue='Bank/App Name')
    plt.title('Rating Distribution by Bank')
    plt.tight_layout()
    plt.savefig('visualizations/rating_distribution.png')
    plt.close()
    
    # 3. Word Cloud for Drivers
    plt.figure(figsize=(10, 6))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(drivers)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Key Drivers (Word Cloud)')
    plt.tight_layout()
    plt.savefig('visualizations/drivers_wordcloud.png')
    plt.close()
    
    # 4. Word Cloud for Pain Points
    plt.figure(figsize=(10, 6))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(pain_points)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Pain Points (Word Cloud)')
    plt.tight_layout()
    plt.savefig('visualizations/pain_points_wordcloud.png')
    plt.close()
    
    # 5. Sentiment Trend Over Ratings
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Rating', y='sentiment_score', hue='Bank/App Name', marker='o')
    plt.title('Sentiment Score Trend by Rating')
    plt.tight_layout()
    plt.savefig('visualizations/sentiment_trend.png')
    plt.close()

def generate_recommendations(drivers: Dict, pain_points: Dict, df: pd.DataFrame) -> List[str]:
    """Generate recommendations based on analysis."""
    recommendations = []
    
    # Analyze common pain points
    if 'slow' in pain_points or 'lag' in pain_points:
        recommendations.append("Optimize app performance and loading times to address user complaints about slowness")
    
    if 'login' in pain_points or 'password' in pain_points:
        recommendations.append("Implement a more user-friendly authentication system with biometric options")
    
    if 'update' in pain_points:
        recommendations.append("Improve update process and communication about new features")
    
    # Analyze positive aspects to reinforce
    if 'easy' in drivers or 'simple' in drivers:
        recommendations.append("Maintain and enhance the user-friendly interface that users appreciate")
    
    if 'service' in drivers or 'support' in drivers:
        recommendations.append("Expand customer support channels and response times")
    
    # Compare banks
    bank_sentiments = df.groupby('Bank/App Name')['sentiment_score'].mean()
    best_practices = bank_sentiments.idxmax()
    recommendations.append(f"Adopt best practices from {best_practices} which shows highest average sentiment")
    
    return recommendations

def analyze_biases(df: pd.DataFrame) -> List[str]:
    """Analyze potential biases in the review data."""
    biases = []
    
    # Check rating distribution
    rating_dist = df['Rating'].value_counts(normalize=True)
    if rating_dist[1] > 0.4:  # If more than 40% are 1-star reviews
        biases.append("Potential negative bias: High concentration of 1-star reviews")
    
    # Check sentiment distribution
    sentiment_dist = df['sentiment_label'].value_counts(normalize=True)
    if sentiment_dist['NEGATIVE'] > 0.6:  # If more than 60% are negative
        biases.append("Potential negative sentiment bias: Majority of reviews are negative")
    
    # Check review length
    df['review_length'] = df['Review Text'].str.len()
    if df['review_length'].mean() < 50:  # If average review is very short
        biases.append("Potential bias: Reviews are generally very short, might not capture full user experience")
    
    return biases

def main():
    print("Loading and analyzing cleaned review data...")
    df = load_cleaned_data()
    
    print("\nAnalyzing drivers and pain points...")
    drivers, pain_points = analyze_drivers_and_pain_points(df)
    
    print("\nCreating visualizations...")
    create_visualizations(df, drivers, pain_points)
    
    print("\nGenerating recommendations...")
    recommendations = generate_recommendations(drivers, pain_points, df)
    
    print("\nAnalyzing potential biases...")
    biases = analyze_biases(df)
    
    # Print insights
    print("\n=== Key Insights ===")
    print("\nTop Drivers:")
    for driver, count in drivers.items():
        print(f"- {driver}: {count} mentions")
    
    print("\nTop Pain Points:")
    for pain, count in pain_points.items():
        print(f"- {pain}: {count} mentions")
    
    print("\nRecommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print("\nPotential Biases:")
    for bias in biases:
        print(f"- {bias}")
    
    print("\nVisualizations have been saved to the 'visualizations' directory")

if __name__ == "__main__":
    main() 