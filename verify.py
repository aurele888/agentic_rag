import logging
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Boolean, JSON
from sqlalchemy.orm import Session

from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path
import os, sys
from sklearn.feature_extraction import text


logger = logging.getLogger(__name__)
# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    # {%(pathname)s:%(lineno)d}
    format='%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join("logs", "app.log"), mode='w'), 
        logging.StreamHandler(sys.stdout)         
    ], 
)


"""Configuration settings for FastAPI application."""
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """Application settings."""

    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    PDF_STORAGE_DIR: str = "data/pdfs/uploads"
    VECTOR_DB_DIR: str = "data/vectors"

    # Database
    DATABASE_URL: str = "sqlite:///./data/api.db"

    # Ollama Config
    OLLAMA_HOST: str = "http://localhost:11434"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    DEFAULT_CHAT_MODEL: str = "qwen3:1.7b"
    DEFAULT_QUERY_ROUTER_MODEL: str = "qwen3:1.7b"

    class Config:
        """Pydantic config."""
        env_file = ".env"

settings = Settings()

# Database configuration
# DATABASE_DIR = Path("/home/aurele/Documents/project/custom-rag/data")
DATABASE_DIR = Path("data")
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_URL = f"sqlite:///{DATABASE_DIR}/api.db"

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Needed for SQLite
    echo=False  # Set to True for SQL debugging
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

class DocumentMetadata(Base):
    """Document metadata table."""
    __tablename__ = "documents"

    pdf_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    collection_name = Column(String, unique=True, nullable=False)
    upload_timestamp = Column(DateTime, nullable=False)
    doc_count = Column(Integer, nullable=False)
    page_count = Column(Integer, nullable=False)
    is_sample = Column(Boolean, default=False)
    file_path = Column(String)

def get_db():
    """Database session dependency."""
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    # db = get_db()
    # logger.info(f"Processing ...")
    # pdfs = []

    # for db in get_db():
    #     query = db.query(DocumentMetadata)
    #     pdfs = query.all()
    #     # data.extend(pdfs)
    #     logger.info(f"Founds {len(pdfs)} pdfs")
      
    # logger.info(f"Data info : {pdfs[0].name}")
    logger.info(f"Keywords: {list(text.ENGLISH_STOP_WORDS)}")
