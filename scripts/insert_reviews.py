from db_connection import Bank, Review, get_session
import pandas as pd

def insert_bank(session, bank_name, country="Ethiopia", website=None):
    """Insert a bank record if it doesn't exist"""
    bank = session.query(Bank).filter_by(name=bank_name).first()
    if not bank:
        bank = Bank(name=bank_name, country=country, website=website)
        session.add(bank)
        session.commit()
    return bank

def insert_reviews(reviews_df):
    """Insert reviews into the database"""
    session = get_session()
    
    try:
        # Create or get banks
        banks = {}
        for bank_name in reviews_df['bank'].unique():
            bank = insert_bank(session, bank_name)
            banks[bank_name] = bank
        
        # Insert reviews
        for _, row in reviews_df.iterrows():
            review = Review(
                bank_id=banks[row['bank']].id,
                review_text=row['review_text'],
                rating=row['rating'] if 'rating' in row else None,
                sentiment_score=row['sentiment_score'] if 'sentiment_score' in row else None,
                sentiment_label=row['sentiment_label'] if 'sentiment_label' in row else None,
                source=row['source'] if 'source' in row else 'google_play'
            )
            session.add(review)
        
        session.commit()
        print(f"Successfully inserted {len(reviews_df)} reviews")
        
    except Exception as e:
        session.rollback()
        print(f"Error inserting reviews: {str(e)}")
        raise
    finally:
        session.close()

def main():
    # Load your cleaned reviews data
    # Adjust the path as needed
    Bank_of_Abyssinia_reviews_cleaned_df = pd.read_csv('data/cleaned_reviews/Bank_of_Abyssinia_reviews_cleaned.csv')
    Bank_of_Abyssinia_reviews_cleaned_df = Bank_of_Abyssinia_reviews_cleaned_df.rename(columns={'Bank/App Name': 'bank'})
    insert_reviews(Bank_of_Abyssinia_reviews_cleaned_df)

    Commercial_Bank_of_Ethiopia_reviews_cleaned_df = pd.read_csv('data/cleaned_reviews/Commercial_Bank_of_Ethiopia_reviews_cleaned.csv')
    Commercial_Bank_of_Ethiopia_reviews_cleaned_df = Commercial_Bank_of_Ethiopia_reviews_cleaned_df.rename(columns={'Bank/App Name': 'bank'})
    insert_reviews(Commercial_Bank_of_Ethiopia_reviews_cleaned_df)

    Dashen_Bank_reviews_cleaned_df = pd.read_csv('data/cleaned_reviews/Dashen_Bank_reviews_cleaned.csv')
    Dashen_Bank_reviews_cleaned_df = Dashen_Bank_reviews_cleaned_df.rename(columns={'Bank/App Name': 'bank'})
    insert_reviews(Dashen_Bank_reviews_cleaned_df)

if __name__ == "__main__":
    main()
