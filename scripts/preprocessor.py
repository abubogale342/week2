from typing import Dict, Any, List
import pandas as pd
from datetime import datetime


class ReviewPreprocessor:
    """A class to handle cleaning and preprocessing of app review data."""
    
    @staticmethod
    def clean_reviews(reviews_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Clean and preprocess review data.
        
        Args:
            reviews_data: List of review dictionaries
            
        Returns:
            Cleaned and preprocessed DataFrame
        """
        if not reviews_data:
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(reviews_data)
        
        # 1. Remove exact duplicates based on reviewId
        df = df.drop_duplicates(subset=['reviewId'], keep='first')
        
        # 2. Handle missing data
        # Fill missing text with empty string
        if 'content' in df.columns:
            df['content'] = df['content'].fillna('')
        
        # Fill missing scores with 0 (neutral)
        if 'score' in df.columns:
            df['score'] = df['score'].fillna(0).astype(int)
            
        # 3. Normalize dates to YYYY-MM-DD format
        if 'at' in df.columns:
            # Convert to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df['at']):
                df['at'] = pd.to_datetime(df['at'])
            # Format as YYYY-MM-DD
            df['at'] = df['at'].dt.strftime('%Y-%m-%d')
        
        # 4. Clean text data (remove extra whitespace, etc.)
        if 'content' in df.columns:
            df['content'] = df['content'].str.strip()
        
        return df
    
    @staticmethod
    def prepare_for_export(df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for export by selecting and renaming columns.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame ready for export
        """
        if df.empty:
            return df
            
        # Select and rename columns
        column_mapping = {
            'app_name': 'Bank/App Name',
            'score': 'Rating',
            'at': 'Date',
            'content': 'Review Text',
            'reviewId': 'Review ID'
        }
        
        # Only include columns that exist in the DataFrame
        columns_to_keep = [col for col in column_mapping.keys() if col in df.columns]
        df = df[columns_to_keep].rename(columns=column_mapping)
        
        # Add source column
        df['Source'] = 'Google Play Store'
        
        return df
