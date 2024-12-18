from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from enum import Enum

import uvicorn
import yaml
from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import Query
from fastapi import UploadFile
from pydantic import BaseModel
from backend.service.file_processing_service import FileProcessingService
from backend.service.query_service import QueryService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class FileType(str, Enum):
    PDF = "pdf"
    SCANNED = "scanned"


class QueryModel(BaseModel):
    text: str
    certainty: float = 0.7


class Document(BaseModel):
    id: int
    text: str


class MainClass:
    def __init__(self):
        self.app = FastAPI(
            swagger_ui_parameters={
                "syntaxHighlight.theme": "obsidian",
            },
        )


    def create_routes(self):
        @self.app.post("/upload")
        async def upload_pdf(
            file: UploadFile = File(...),
            file_type: FileType = FileType.PDF,
            start_page: int = Query(
                0,
                description="The starting page for extraction.",
            ),
            end_page: int = Query(
                default=None,
                description="The ending page for extraction.",
            ),
            min_word: int = Query(
                512,
                description="Min word count per record.",
            ),
        ):
            try:
                self.file_processing_service = FileProcessingService(
                    self.weaviate_instance,
                )
                filtered_text = self.file_processing_service.process_file(
                    file,
                    file_type.value,
                    start_page=start_page,
                    end_page=end_page,
                    min_word=min_word,
                )

                self.file_processing_service.embedding_data(filtered_text)
                # self.worker_update.update_laws(filtered_text)

                return {"message": "File processed and embedded successfully"}
            except Exception as e:
                logging.error(f"Error in /upload endpoint: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Internal Server Error: {str(e)}",
                )

        @self.app.post("/search")
        async def search_documents(request: QueryModel):
            try:
                self.query_service = QueryService(self.weaviate_instance)
                return self.query_service.get_data(request)
            except Exception as e:
                logging.error(f"Error in /search endpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/update")
        async def update_pdf(
            file: UploadFile = File(...),
            file_type: FileType = FileType.PDF,
            start_page: int = Query(
                0,
                description="The starting page for extraction.",
            ),
            end_page: int = Query(
                default=None,
                description="The ending page for extraction.",
            ),
            min_word: int = Query(
                512,
                description="Min word count per record.",
            ),
        ):
            try:
                self.file_processing_service = FileProcessingService(
                    self.weaviate_instance,
                )
                filtered_text = self.file_processing_service.process_file(
                    file,
                    file_type.value,
                    start_page=start_page,
                    end_page=end_page,
                    min_word=min_word,
                )

                self.worker_update.update_laws(filtered_text)

                return {"message": "File processed and embedded successfully"}
            except Exception as e:
                logging.error(f"Error in /upload endpoint: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Internal Server Error: {str(e)}",
                )

        @self.app.post("/ask_question")
        async def ask_question(request: QueryModel):
            try:
                self.query_service = QueryService(self.weaviate_instance)
                return self.query_service.ask_question(request)
            except Exception as e:
                logging.error(f"Error in /ask endpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def run(self, host="0.0.0.0", port=8000):
        # Chạy ứng dụng với uvicorn
        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
    main_app = MainClass()
    main_app.create_routes()
    main_app.run()
