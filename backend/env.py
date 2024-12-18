from dotenv import load_dotenv, find_dotenv
import os
from pathlib import Path

# Define paths
BACKEND_DIR = Path(__file__).parent
BASE_DIR = BACKEND_DIR.parent

# Attempt to load .env file
dotenv_path = find_dotenv(str(BASE_DIR / ".env"))
if dotenv_path:
    load_dotenv(dotenv_path)
    print("Loaded .env file successfully")
else:
    print(f"Could not find .env file at {BASE_DIR / '.env'}")

# Get environment variables
ENV = os.environ.get("ENV", "dev")
VECTOR_DB = os.environ.get("VECTOR_DB", "weaviate")
WEAVIATE_URL = os.environ.get("WEAVIATE_URL")
WEAVIATE_GRPC_URL = os.environ.get("WEAVIATE_GRPC_URL")
WEAVIATE_PREFER_GRPC = os.environ.get("WEAVIATE_PREFER_GRPC", "False").lower() in ("true", "1", "t")

