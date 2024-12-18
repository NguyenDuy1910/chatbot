from __future__ import annotations

import os

import dotenv

dotenv.load_dotenv()
STORAGE_PROVIDER = os.getenv("STORAGE_PROVIDER", "local")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "your_default_access_key_id")
S3_SECRET_ACCESS_KEY = os.getenv(
    "S3_SECRET_ACCESS_KEY",
    "your_default_secret_access_key",
)
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "your_default_bucket_name")
S3_REGION_NAME = os.getenv("S3_REGION_NAME", "your_default_region_name")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "your_default_endpoint_url")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads/")

# Export environment variables
if __name__ == "__main__":
    print(f"export STORAGE_PROVIDER={STORAGE_PROVIDER}")
    print(f"export S3_ACCESS_KEY_ID={S3_ACCESS_KEY_ID}")
    print(f"export S3_SECRET_ACCESS_KEY={S3_SECRET_ACCESS_KEY}")
    print(f"export S3_BUCKET_NAME={S3_BUCKET_NAME}")
    print(f"export S3_REGION_NAME={S3_REGION_NAME}")
    print(f"export S3_ENDPOINT_URL={S3_ENDPOINT_URL}")
    print(f"export UPLOAD_DIR={UPLOAD_DIR}")
