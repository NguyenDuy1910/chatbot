import os
from backend.env import (
    VECTOR_DB,
    WEAVIATE_URL,
    WEAVIATE_GRPC_URL,
    WEAVIATE_PREFER_GRPC,
)
import weaviate
import weaviate.classes as wvc

# Define the base directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Database Configuration
DATABASE_URL = "sqlite:///example.db"
DATABASE_POOL_SIZE = 5
DATABASE_MAX_OVERFLOW = 10
DATABASE_POOL_RECYCLE = 3600
DATABASE_POOL_TIMEOUT = 30

# Weaviate Configuration
WEAVIATE_COLLECTION = "Document"
WEAVIATE_OVERWRITE_INDEX = False

# Correct boolean type for WEAVIATE_PREFER_GRPC if using os.environ.get() fallback
if isinstance(WEAVIATE_PREFER_GRPC, str):
    WEAVIATE_PREFER_GRPC = WEAVIATE_PREFER_GRPC.lower() in ("true", "1", "t")

# Weaviate Schema Definition
WEAVIATE_SCHEMA = {
    "name": "LegalDocumentCollection",
    "vectorizer_config": wvc.config.Configure.Vectorizer.none(),
    "generative_config": wvc.config.Configure.Generative.cohere(),
    "description": "A collection of legal documents",
    "properties": [
        wvc.config.Property(
            name="doc_id",
            data_type=wvc.config.DataType.INT,  # Updated to Weaviate data type
            description="Unique identifier for the document",
        ),
        wvc.config.Property(
            name="text",
            data_type=wvc.config.DataType.TEXT,  # Updated to Weaviate data type
            description="The content of the legal document",
        ),
        wvc.config.Property(
            name="law_number",
            data_type=wvc.config.DataType.INT,  # Updated to Weaviate data type
            description="The associated law number",
        ),
    ],
    "vector_index_config": {
        "distance": "cosine",
        "efConstruction": 128,  # Optional: config for building the HNSW index
        "maxConnections": 64,  # Optional: number of connections for the index
    }
}

# Weaviate Search Parameters
WEAVIATE_SEARCH_PARAMS = {
    "hnsw_ef": 512,
    "ef_search": 512,
}

# Logging Levels
LOG_LEVELS = {
    "COMFYUI": "DEBUG",
    "DATABASE": "INFO",
    "SERVER": "ERROR",
}
