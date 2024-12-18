from __future__ import annotations

import logging
import os

from fastapi import UploadFile
from typing import List, Dict, Any
from txtai.embeddings import Embeddings
from backend.app.vector.dbs.weaviate_db import WeaviateClient
from backend.data.pdf.pdf_processing import extract_text_from_pdf, extract_laws_from_text
from backend.data.pdf.pdf_processing import perform_ocr_on_directory
from backend.data.pdf.pdf_processing import convert_pdf_to_images
from backend.data.pdf.pdf_processing import filter_laws_by_length

BASE_STORAGE = "/Users/nguyendinhquocduy/Documents/techJDI-training/build-chatbot/backend/storage/uploads"


class FileProcessingService:
    def __init__(self, weaviate_instance: WeaviateClient):
        """
        Initialize FileProcessingService with an instance of Weaviate.
        :param weaviate_instance: The initialized instance of Weaviate.
        """
        self.weaviate_instance = weaviate_instance
        # Initialize Embeddings using txtai
        self.embeddings = Embeddings(
            {"path": "sentence-transformers/all-MiniLM-L6-v2"},
        )

    def save_uploaded_file(self, file: UploadFile, base_storage: str) -> str:
        """Save the uploaded file to a specified location if it doesn't already exist."""
        try:
            os.makedirs(base_storage, exist_ok=True)

            file_location = os.path.join(base_storage, file.filename)
            if os.path.exists(file_location):
                logging.info(f"File already exists at {file_location}")
                return file_location
            with open(file_location, "wb") as f:
                f.write(file.file.read())

            logging.info(f"File saved to {file_location}")
            return file_location

        except Exception as e:
            logging.error(f"Failed to save file: {e}")
            raise RuntimeError("Error saving file")

    def process_file(
            self,
            file: UploadFile,
            file_type: str,
            start_page: int,
            end_page: int,
            min_word: int,
    ) -> Dict[str, str]:
        try:
            # Save the uploaded file
            file_location = self.save_uploaded_file(file, BASE_STORAGE)

            # Extract text based on file type
            if file_type == "pdf":
                extracted_text = extract_text_from_pdf(
                    file_location,
                    start_page=start_page,
                    end_page=end_page,
                )
            else:
                folder_output = convert_pdf_to_images(
                    file_location,
                    start_page=start_page,
                    end_page=end_page,
                )
                extracted_text = perform_ocr_on_directory(
                    provider="local",
                    output_dir=folder_output,
                )

            if not extracted_text:
                raise ValueError("No text could be extracted from the file")

            # Split and process content
            split_content = extract_laws_from_text(extracted_text)
            if not split_content:
                raise ValueError(
                    "Failed to split content from the extracted text",
                )

            filtered_text = filter_laws_by_length(split_content, min_word)
            logging.info("Filtered extracted text based on token count.")
            return filtered_text

        except Exception as e:
            logging.error(f"Error during file processing: {e}")
            raise RuntimeError(f"Error processing file: {e}")

    def embedding_data(self, filtered_text: Dict[str, str]):
        try:
            if not filtered_text:
                raise ValueError("No text provided for embedding")
            logging.info("Successfully indexed the data into txtai.")

            # Generate embeddings and prepare data for Weaviate
            vectors = []
            objects = []
            for law_number, content in filtered_text.items():
                if content.strip():  # Skip empty entries
                    embedding = self.embeddings.transform([content])
                    obj = {
                        'law_number': law_number,
                        'text': content,
                    }
                    vectors.append(embedding)
                    objects.append(obj)

            # Save data to Weaviate
            self.save_data(objects, vectors)
        except Exception as e:
            logging.error(f"Error during embedding data: {e}")
            raise RuntimeError(f"Error embedding data: {e}")

    def save_data(self, objects: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Save embeddings and content to Weaviate.
        :param objects: List of dictionaries containing the object data.
        :param embeddings: List of embedding vectors.
        """
        try:
            if not embeddings or not objects:
                raise ValueError("No embeddings or objects provided for saving.")

            # Insert objects and their embeddings into Weaviate
            self.weaviate_instance.insert_objects(
                class_name="LegalDocumentCollection",
                objects=[
                    {
                        "law_number": obj["law_number"],
                        "text": obj["text"]
                    }
                    for obj in objects
                ],
                vectors=embeddings
            )
            logging.info("Successfully appended embeddings to Weaviate.")

        except Exception as e:
            logging.error(f"Failed to save data to Weaviate: {e}")
            raise RuntimeError(f"Failed to save data: {e}")
