from __future__ import annotations

import weaviate
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from transformers import AutoModel
from transformers import AutoTokenizer

# FastAPI application instance
app = FastAPI()

# Connect to Weaviate
client = weaviate.Client("http://localhost:8080")
if not client.is_ready():
    raise Exception("Weaviate connection failed. Check the server.")

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
)
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


# Input model for the query
class QueryRequest(BaseModel):
    query: str
    certainty: float = 0.73  # Default certainty threshold


@app.delete("/schema")
async def delete_schema():
    """
    Deletes the entire schema from Weaviate, including all classes and objects.
    """
    try:
        # Delete all schema
        client.schema.delete_all()
        return {
            "message": "All schemas and associated data have been deleted successfully.",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete schema: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
