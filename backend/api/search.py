# import logging
# from typing import List, Dict, Any
# import uvicorn
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
#
# from backend.app.vector.dbs.weaviate_db import WeaviateClient
# from txtai.embeddings import Embeddings
#
# from backend.core.query.hyde import HyDE
#
# # Initialize FastAPI app
# app = FastAPI()
#
# # Initialize WeaviateClient
# weaviate_client = WeaviateClient()
#
# # Initialize txtai embeddings
# embeddings = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})
#
# # Request model for search endpoint
# class SearchRequest(BaseModel):
#     class_name: str
#     query: str
#     limit: int = 10
#
# # Response model for search endpoint
# class SearchResult(BaseModel):
#     doc_id: int
#     text: str
#     law_number: int
#
# @app.post("/search", response_model=List[SearchResult])
# async def search_endpoint(request: SearchRequest):
#     """
#     Search endpoint for querying Weaviate.
#     """
#     try:
#         # Generate embedding for the input query
#         query_vector = embeddings.transform([request.query])
#
#         # Perform search using WeaviateClient
#         results = weaviate_client.search(
#             class_name=request.class_name,
#             vector=query_vector,
#             limit=request.limit
#         )
#
#         # Format results
#         formatted_results = [
#             {
#                 "doc_id": result["doc_id"],
#                 "text": result["text"],
#                 "law_number": result["law_number"],
#             }
#             for result in results
#         ]
#
#         return formatted_results
#     except RuntimeError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         logging.error(f"Unexpected error: {e}")
#         raise HTTPException(status_code=500, detail="An unexpected error occurred.")
#
#
# def main():
#     """
#     Entry point to start the FastAPI app.
#     """
#     logging.basicConfig(level=logging.INFO)
#     uvicorn.run("search:app", host="127.0.0.1", port=8000, reload=True)
#
# # Run the main function if this script is executed directly
# if __name__ == "__main__":
#     main()
import logging
import os
from typing import List, Any

import dotenv
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from backend.app.vector.dbs.weaviate_db import WeaviateClient
from backend.core.query.hyde import HyDE, Promptor, OpenAIGenerator

# Load environment variables
dotenv.load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize WeaviateClient
weaviate_client = WeaviateClient()

# Initialize SentenceTransformer embeddings
encoder = SentenceTransformer("all-MiniLM-L6-v2")


promptor = Promptor(task="legal advice", language="vi")
generator = OpenAIGenerator(
    model_name="gpt-3.5-turbo",
    api_key=os.getenv("OPEN_API_KEY"),
    n=8,
    max_tokens=512,
    temperature=0.7,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.0,
)

# Initialize HyDE
hyde = HyDE(promptor, generator, encoder, weaviate_client)


# Request model for search endpoint
class SearchRequest(BaseModel):
    class_name: str
    query: str
    limit: int = 10


# Response model for search endpoint
class SearchResult(BaseModel):
    doc_id: int
    text: str
    law_number: int


@app.post("/search", response_model=Any)
async def search_endpoint(request: SearchRequest):
    """
    Search endpoint for querying Weaviate.
    """
    try:
        logging.info(f"Received search request for class: {request.class_name}")

        # Perform HyDE process to generate embeddings and query Weaviate
        logging.info("Building prompt...")
        prompt = hyde.prompt(request.query)

        logging.info("Generating hypothetical documents...")
        hypothetical_documents = hyde.generate(request.query)

        logging.info("Encoding query and hypothesis documents...")
        hyde_vector = hyde.encode(request.query, hypothetical_documents)

        logging.info("Performing search...")
        results = weaviate_client.search(
            class_name=request.class_name,
            vector=hyde_vector,
            limit=request.limit
        )
        return results
        # formatted_results = [
        #     {
        #         "doc_id": result["doc_id"],
        #         "text": result["text"],
        #         "law_number": result["law_number"],
        #     }
        #     for result in results
        # ]
        #
        # # Format results
        #
        #
        # return formatted_results
    except ValueError as e:
        logging.error(f"Value error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logging.error(f"Runtime error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


def main():
    """
    Entry point to start the FastAPI app.
    """
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("search:app", host="127.0.0.1", port=8000, reload=True)


# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()
