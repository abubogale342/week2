from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

from dotenv import load_dotenv
import os

load_dotenv()

class Bank(Base):
    __tablename__ = 'banks'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    country = Column(String(50))
    website = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Review(Base):
    __tablename__ = 'reviews'
    
    id = Column(Integer, primary_key=True)
    bank_id = Column(Integer, nullable=False)
    review_text = Column(Text, nullable=False)
    rating = Column(Float)
    sentiment_score = Column(Float)
    sentiment_label = Column(String(20))
    source = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

def get_db_engine():
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")

    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(DATABASE_URL)
    return engine

def init_db():
    """Initialize the database and create tables if they don't exist"""
    engine = get_db_engine()
    Base.metadata.create_all(engine)
    return engine

def get_session():
    """Get a database session"""
    engine = get_db_engine()
    Session = sessionmaker(bind=engine)
    session = Session()
    return session
