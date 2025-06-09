from google_play_scraper import app, Sort, reviews
import pandas as pd
from datetime import datetime
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json

try:
    from scripts.preprocessor import ReviewPreprocessor
except ImportError:
    from preprocessor import ReviewPreprocessor


class PlayStoreScraper:
    """A class to scrape app reviews from Google Play Store."""

    def __init__(self, output_dir: str = 'data') -> None:
        """Initialize the scraper with default settings.
        
        Args:
            output_dir: Directory to save scraped data
        """
        self.apps: Dict[str, str] = {
            "Commercial Bank of Ethiopia": "com.combanketh.mobilebanking",
            "Dashen Bank": "com.dashen.dashensuperapp",
            "Bank of Abyssinia": "com.boa.boaMobileBanking"
        }
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_app_reviews(self, app_name: str, app_id: str, count: int = 400) -> List[Dict[str, Any]]:
        """Fetch reviews for a specific app from Google Play Store.
        
        Args:
            app_name: Name of the bank/app
            app_id: Google Play Store app ID
            count: Number of reviews to fetch (default: 400)
        
        Returns:
            List of dictionaries containing review data
        """
        print(f"Fetching reviews for {app_name}...")
        
        try:
            # Get app info
            app_info = app(app_id)
            
            all_reviews = []
            continuation_token = None
            
            while len(all_reviews) < count:
                remaining = count - len(all_reviews)
                batch_size = min(200, remaining)
                
                result, continuation_token = reviews(
                    app_id,
                    lang='en',
                    country='et',
                    sort=Sort.NEWEST,
                    count=batch_size,
                    continuation_token=continuation_token
                )
                
                if not result:
                    break
                    
                # Add app info to each review
                for review in result:
                    review['app_name'] = app_name
                    review['app_id'] = app_id
                
                all_reviews.extend(result)
                
                if continuation_token is None or len(all_reviews) >= count:
                    break
                    
                time.sleep(2)  # Be nice to Google's servers
            
            print(f"Fetched {len(all_reviews)} reviews for {app_name}")
            return all_reviews
        
        except Exception as e:
            print(f"Error fetching reviews for {app_name}: {str(e)}")
            return []

    def save_reviews(self, reviews_data: List[Dict[str, Any]], app_name: str) -> tuple[str, str]:
        """Save reviews to CSV and JSON files after preprocessing.
        
        Args:
            reviews_data: List of review dictionaries
            app_name: Name of the app (used for filename)
            
        Returns:
            Tuple of (csv_path, json_path) where files were saved
        """
        if not reviews_data:
            print("No reviews to save.")
            return "", ""
        
        # Sanitize app name for filename
        safe_app_name = "".join(c if c.isalnum() else "_" for c in app_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Clean and preprocess the data
            preprocessor = ReviewPreprocessor()
            cleaned_df = preprocessor.clean_reviews(reviews_data)
            
            if cleaned_df.empty:
                print("No valid reviews to save after preprocessing.")
                return "", ""
                
            # Prepare for export
            export_df = preprocessor.prepare_for_export(cleaned_df)
            
            # Generate file paths
            base_filename = f"{safe_app_name}_reviews_{timestamp}"
            csv_path = self.output_dir / f"{base_filename}.csv"
            json_path = self.output_dir / f"{base_filename}.json"
            
            # Save to CSV
            export_df.to_csv(csv_path, index=False)
            print(f"Saved {len(export_df)} reviews to {csv_path}")
            
            return str(csv_path), ""
            
        except Exception as e:
            print(f"Error saving reviews for {app_name}: {str(e)}")
            return "", ""

    def scrape_all(self, reviews_per_app: int = 400) -> Dict[str, List[Dict[str, Any]]]:
        """Scrape reviews for all configured apps.
        
        Args:
            reviews_per_app: Number of reviews to fetch per app
            
        Returns:
            Dictionary mapping app names to their reviews
        """
        all_reviews = {}
        
        for app_name, app_id in self.apps.items():
            reviews = self.get_app_reviews(app_name, app_id, reviews_per_app)
            self.save_reviews(reviews, app_name)
            all_reviews[app_name] = reviews
            
        print(f"Total reviews collected: {sum(len(v) for v in all_reviews.values())}")
        return all_reviews


def main():
    """Main function to demonstrate the scraper usage."""
    scraper = PlayStoreScraper(output_dir='data')
    scraper.scrape_all(reviews_per_app=400)


if __name__ == "__main__":
    main()
