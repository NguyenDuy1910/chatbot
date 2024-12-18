from __future__ import annotations

import glob
import logging
import os
import re
from datetime import datetime
from threading import Timer
from typing import BinaryIO

import pytesseract
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text as pdfminer_extract_text
from PIL import Image
from pypdf import PdfReader

from backend.data.shared.text_processing import text_preprocess
from backend.storage.provider import StorageProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Constants
BASE_STORAGE_DIR = "Retrieve"  # Local storage directory for output files (not used for production)
STORAGE_PROVIDER = StorageProvider(provider="local")  # Initialize storage provider (e.g., S3)


# Utility Functions
def create_output_directory(pdf_file: str) -> str:
    """Create an output directory for the PDF file with a timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join(
        BASE_STORAGE_DIR,
        f"{os.path.splitext(os.path.basename(pdf_file))[0]}_{timestamp}",
    )
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Created output directory: {output_dir}")
    return output_dir


def schedule_directory_deletion(output_dir: str, timeout_minutes: int = 10) -> None:
    """Schedule deletion of the output directory after a specified timeout."""

    def delete_directory():
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, file))
            os.rmdir(output_dir)
            logging.info(f"Deleted temporary directory: {output_dir}")

    Timer(timeout_minutes * 60, delete_directory).start()
    logging.info(f"Scheduled deletion of directory '{output_dir}' in {timeout_minutes} minutes.")


def is_pdf_scanned(pdf_file: str) -> bool:
    """Check if a PDF is scanned by detecting embedded text."""
    text = pdfminer_extract_text(pdf_file)
    scanned = not bool(text.strip())
    logging.info(f"PDF '{pdf_file}' is {'scanned' if scanned else 'not scanned'}.")
    return scanned


# Text Extraction Functions
def extract_text_from_pdf(pdf_file: str, start_page: int = None, end_page: int = None) -> str:
    """Extract text from a PDF using pdfminer."""
    text = pdfminer_extract_text(
        pdf_file,
        page_numbers=range(start_page, end_page) if start_page and end_page else None,
    )
    logging.info(f"Extracted text from PDF '{pdf_file}' from page {start_page} to {end_page}.")
    return text


def extract_text_by_page(pdf_file: str, start_page: int, end_page: int = None) -> list[str]:
    """Extract text from specific pages of a PDF."""
    reader = PdfReader(pdf_file)
    total_pages = len(reader.pages)
    end_page = end_page or total_pages

    if start_page < 1 or end_page < start_page:
        raise ValueError("Invalid page range.")

    page_texts = []
    for page_num in range(start_page, end_page + 1):
        try:
            text = pdfminer_extract_text(pdf_file, page_numbers=[page_num - 1])
            page_texts.append(text)
        except Exception as e:
            page_texts.append(f"Error extracting page {page_num}: {e}")

    return page_texts


def convert_pdf_to_images(pdf_file: str, start_page: int = 1, end_page: int = None) -> str:
    """Convert PDF pages to images."""
    output_dir = create_output_directory(pdf_file)
    pages = convert_from_path(
        pdf_file,
        dpi=500,
        first_page=start_page,
        last_page=end_page,
    )
    for idx, page in enumerate(pages, start=start_page):
        image_path = os.path.join(output_dir, f"page_{idx}.jpg")
        page.save(image_path, "JPEG")
        logging.info(f"Saved page {idx} of PDF '{pdf_file}' as image: {image_path}")

    schedule_directory_deletion(output_dir)
    return output_dir


# OCR Functions
def perform_ocr_on_image(image_path: str) -> str:
    """Perform OCR on a single image."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang="vie", config="--psm 6")
    logging.info(f"Performed OCR on image: {image_path}")
    return text


def perform_ocr_on_directory(provider: str, output_dir: str) -> str:
    """Perform OCR on all images in a directory."""
    if provider == "s3":
        image_paths = sorted(
            STORAGE_PROVIDER.get_file(f"{output_dir}/"),
            key=lambda x: int(re.search(r"page_(\d+).jpg", x).group(1)),
        )
    else:  # Local provider
        image_paths = sorted(
            glob.glob(f"{output_dir}/*.jpg"),
            key=lambda x: int(re.search(r"page_(\d+).jpg", x).group(1)),
        )

    ocr_results = []
    for image_path in image_paths:
        local_path = STORAGE_PROVIDER.get_file(image_path) if provider == "s3" else image_path
        ocr_results.append(perform_ocr_on_image(local_path))
        if provider == "s3":
            os.remove(local_path)

    logging.info(f"Completed OCR on all images in directory: {output_dir}")
    return "\n".join(ocr_results)


# Post-OCR Processing
def extract_laws_from_text(text: str) -> dict[int, str]:
    """Extract laws based on pattern matching."""
    try:
        pattern = r"(Điều \d+\..*?)(?=(Điều \d+\.|\Z))"
        matches = re.finditer(pattern, text, flags=re.DOTALL)

        law_dict = {}
        for match in matches:
            law_number = int(re.match(r"Điều (\d+)", match.group(0)).group(1))
            content = match.group(0).strip()
            law_dict[law_number] = content

        logging.info("Extracted laws from text.")
        return law_dict
    except Exception as e:
        logging.error(f"Error extracting laws: {e}")
        return {}


def count_words_in_text(text: str) -> int:
    """Count the number of words in a text."""
    return len(text.split()) if isinstance(text, str) else 0


def filter_laws_by_length(laws: dict[int, str], min_length: int) -> dict[int, str]:
    """Filter laws by minimum word count."""
    filtered = {
        number: text_preprocess(content)
        for number, content in laws.items()
        if count_words_in_text(content) >= min_length
    }
    logging.info(f"Filtered laws with at least {min_length} words.")
    return filtered


# File Upload
def upload_file(file: BinaryIO, filename: str) -> tuple[bytes, str]:
    """Upload a file to S3 or local storage."""
    contents = file.read()
    if not contents:
        raise ValueError("File is empty.")

    local_path = "temporary"
    contents, local_path = STORAGE_PROVIDER.upload_file(local_path, filename)
    logging.info(f"Uploaded file to local storage: {local_path}")

    return contents, local_path
