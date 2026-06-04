"""Configuration settings for FastAPI application."""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List, Set

class Settings(BaseSettings):
    """Application settings."""

    # General Config
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    PDF_STORAGE_DIR: str = "data/pdfs/uploads"
    VECTOR_DB_DIR: str = "data/vectors"
    SUPPORTED_FILES_EXTENSION : List[str] = ['.pdf', '.docx', '.pptx', '.csv']
    LANGUAGE : str = "english"

    # Database
    DATABASE_URL: str = "sqlite:///./data/api.db"

    # Ollama Config
    OLLAMA_HOST: str = "http://localhost:11434"
    EMBEDDING_MODEL: str = "nomic-embed-text"

    # Chat model config
    DEFAULT_CHAT_MODEL: str = "qwen3:1.7b"
    CHAT_MODEL_TEMPERATURE: float = 0.7
    CHAT_MODEL_TOP_P: float = 0.95
    CHAT_MODEL_TOP_K: int = 20
    
    # Query router config
    DEFAULT_QUERY_ROUTER_MODEL: str = "qwen3:1.7b"
    DEFAULT_QUERY_ROUTER_THRESHOLD: float = 0.0
    DEFAULT_QUERY_ROUTER_MAX_WORKERS: int = 4
    DEFAULT_QUERY_ROUTER_TEMPERATURE: float = 0.6
    DEFAULT_QUERY_ROUTER_TOP_P: float = 0.95
    DEFAULT_QUERY_ROUTER_TOP_K: int = 20

    # RAG config
    USE_HYBRID_SEARCH : bool = True
    KEYWORDS_SIMILARITY_WEIGHT: float = 0.35
    SEMANTIC_SIMILARITY_WEIGHT: float = 0.65
    TOP_K_CHUNK: int = 1000
    USE_RE_RANKER: bool = False
    RE_RANKER_THRESHOLD: float = 0.65
    RE_RANKER_MODEL : str = "zeroentropy/zerank-1-small"

    # Document indexing config
    KEYWORD_THRESHOLD_SCORE : float = 0.35
    DEFAULT_SENTENCE_TRANSFORMER_MODEL : str = "all-MiniLM-L6-v2"

    
    class Config:
        """Pydantic config."""
        env_file = ".env"

settings = Settings()