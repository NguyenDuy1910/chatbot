from __future__ import annotations

import os
import shutil

import boto3
from botocore.exceptions import ClientError

from config import (
    S3_ACCESS_KEY_ID,
    S3_BUCKET_NAME,
    S3_ENDPOINT_URL,
    S3_REGION_NAME,
    S3_SECRET_ACCESS_KEY,
    STORAGE_PROVIDER,
    UPLOAD_DIR,
)


class StorageProvider:
    def __init__(self, provider: str | None = None):
        self.storage_provider: str = provider or STORAGE_PROVIDER
        self.s3_client = None
        self.bucket_name: str | None = None

        if self.storage_provider == "s3":
            self._initialize_s3()

    def _initialize_s3(self) -> None:
        """Initializes the S3 client and bucket name if using S3 storage."""
        self.s3_client = boto3.client(
            "s3",
            region_name=S3_REGION_NAME,
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=S3_ACCESS_KEY_ID,
            aws_secret_access_key=S3_SECRET_ACCESS_KEY,
        )
        self.bucket_name = S3_BUCKET_NAME

    def _ensure_s3_client_initialized(self):
        if not self.s3_client:
            raise RuntimeError("S3 Client is not initialized.")

    def _upload_to_s3(self, filename: str, object_name: str | None = None) -> tuple[bytes, str]:
        """Uploads a file to S3 storage."""
        self._ensure_s3_client_initialized()

        object_name = object_name or os.path.basename(filename)
        try:
            self.s3_client.upload_file(filename, self.bucket_name, object_name)
            with open(filename, "rb") as f:
                return f.read(), filename
        except ClientError as e:
            raise RuntimeError(f"Error uploading file to S3: {e}")

    def _get_file_from_s3(self, file_path: str) -> str:
        """Downloads a file from S3 storage."""
        self._ensure_s3_client_initialized()

        key = file_path.split(f"{self.bucket_name}/")[-1]
        local_file_path = os.path.join(UPLOAD_DIR, key)
        try:
            self.s3_client.download_file(self.bucket_name, key, local_file_path)
            return local_file_path
        except ClientError as e:
            raise RuntimeError(f"Error downloading file from S3: {e}")

    def _delete_from_s3(self, filename: str) -> None:
        """Deletes a file from S3 storage."""
        self._ensure_s3_client_initialized()

        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=filename)
        except ClientError as e:
            raise RuntimeError(f"Error deleting file from S3: {e}")

    def _delete_all_from_s3(self) -> None:
        """Deletes all files from S3 storage."""
        self._ensure_s3_client_initialized()

        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            for content in response.get("Contents", []):
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=content["Key"])
        except ClientError as e:
            raise RuntimeError(f"Error deleting all files from S3: {e}")
    @staticmethod
    def _upload_to_local(file_path: str, filename: str) -> tuple[bytes, str]:
        """Uploads a file to local storage."""
        dest_path = os.path.join(UPLOAD_DIR, filename)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        shutil.copy(file_path, dest_path)
        with open(dest_path, "rb") as f:
            return f.read(), dest_path

    @staticmethod
    def _get_file_from_local(file_path: str) -> str:
        """Returns the local file path."""
        return file_path

    @staticmethod
    def _delete_from_local(filename: str) -> None:
        """Deletes a file from local storage."""
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    @staticmethod
    def _delete_all_from_local() -> None:
        """Deletes all files from local storage."""
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)

    def upload_file(self, filename: str, file_path: str) -> tuple[bytes, str]:
        """Uploads a file to the appropriate storage."""
        if self.storage_provider == "s3":
            return self._upload_to_s3(filename, file_path)
        return self._upload_to_local(file_path, filename)

    def get_file(self, file_path: str) -> str:
        """Downloads a file from the appropriate storage."""
        if self.storage_provider == "s3":
            return self._get_file_from_s3(file_path)
        return self._get_file_from_local(file_path)

    def delete_file(self, filename: str) -> None:
        """Deletes a file from the appropriate storage."""
        if self.storage_provider == "s3":
            self._delete_from_s3(filename)
        self._delete_from_local(filename)

    def delete_all_files(self) -> None:
        try:
            if self.storage_provider == "s3":
                self._delete_all_from_s3()
            self._delete_all_from_local()
        except Exception as e:
            raise RuntimeError(f"Failed to delete all files: {e}")

