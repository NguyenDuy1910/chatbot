from __future__ import annotations

import logging
import os

import dotenv
from fastapi import HTTPException
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from txtai.embeddings import Embeddings

from backend.app.vector.dbs.weaviate_db import WeaviateClient

BASE_STORAGE = "/Users/nguyendinhquocduy/Documents/techJDI-training/build-chatbot/backend/storage/uploads"
dotenv.load_dotenv()


class QueryModel(BaseModel):
    text: str
    certainty: float = 0.7  # Default certainty


class QueryService:
    def __init__(self, weaviate_instance: WeaviateClient):
        self.llm = ChatOpenAI(
            api_key=os.getenv(
                "OPEN_API_KEY",
            ),
            model_name="gpt-3.5-turbo",
            temperature=0.7,
        )
        self.weaviate_instance = weaviate_instance
        self.embeddings = Embeddings(
            {"path": "sentence-transformers/all-MiniLM-L6-v2"},
        )

    def get_data(self, query: QueryModel):
        try:
            # Set a default value for certainty if it's None
            if query.certainty is None:
                query.certainty = 0.5

            # Generate query vector using embeddings
            query_vector = self.embeddings.transform(query.text)

            # Query Weaviate with the vector
            result = (
                self.weaviate_instance.client.query.get(
                    "Document",
                    ["docid", "law_number", "text"],
                )
                .with_near_vector(
                    {
                        "vector": query_vector.tolist(),
                        "certainty": query.certainty,
                    },
                )
                .with_additional("distance")
                .do()
            )

            # Process results
            if (
                result
                and "data" in result
                and "Get" in result["data"]
                and "Document" in result["data"]["Get"]
            ):
                documents = result["data"]["Get"]["Document"]
                if documents:
                    # Process documents and calculate similarity scores
                    processed_docs = [
                        {
                            "text": doc["text"],
                            "docid": doc.get("docid"),
                            "law_number": doc.get("law_number"),
                            # Convert distance to similarity score
                            "score": 1 - doc["_additional"]["distance"],
                        }
                        for doc in documents
                    ]
                    top_docs = sorted(
                        processed_docs,
                        key=lambda x: x["score"],
                        reverse=True,
                    )[:3]

                    return {"results": top_docs}
                else:
                    return {"results": []}
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Query failed or returned no data.",
                )
        except Exception as e:
            logging.error(f"Error in get_data method: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def ask_question(self, query):
        try:
            # Step 1: Retrieve data from your existing logic or database
            # Assuming this fetches relevant data or context
            context_data = self.get_data(query)

            # Step 2: Define the prompt for the AI assistant
            search_prompt = PromptTemplate(
                input_variables=["query", "context"],
                template="""
                Bạn là một trợ lý AI hỗ trợ trả lời các điều luật cho người dùng. Người dùng đã đưa ra yêu cầu sau:
                {query}

                Ngữ cảnh dữ liệu có sẵn (nếu có): {context}

                Nhiệm vụ của bạn:
                1. Hiểu rõ mục đích của người dùng.
                2. Tìm kiếm các khái niệm hoặc đối tượng liên quan dựa trên yêu cầu.
                3. Cung cấp một bản tóm tắt có cấu trúc về các kết quả hoặc thông tin liên quan.
                4. Bạn không được làm giả thông tin mà phải dựa vào context đã cung cấp. Hãy trả lời và đưa ra các gợi ý cho người dùng hỏi tiếp dựa vào ngữ cảnh nếu bạn không có đủ thông tin.

                Trả lời bằng định dạng JSON có cấu trúc:
                {{
                  "intent": "Mục đích đã được xác định của người dùng",
                  "results": [
                    {{
                      "name": "Tên kết quả có thể phù hợp",
                      "description": "Mô tả ngắn gọn về kết quả",
                      "relevance": "Cao/Trung bình/Thấp"
                    }}
                  ],
                  "recommendation": "Các gợi ý hoặc bước tiếp theo cho người dùng"
                }}
                """,
            )
            chain = search_prompt | self.llm
            response = chain.invoke({"query": query, "context": context_data})

            return response

        except Exception as e:
            logging.error(f"Error in search_documents: {e}")
            raise HTTPException(
                status_code=500,
                detail="Internal Server Error",
            )

    def update_data(self, query: QueryModel):
        """
        Delete the most similar record in the database and insert a new one.
        """
        try:
            # Generate embedding for the new text
            new_vector = self.embeddings.transform(query.text)

            # Query Weaviate to find the most similar document
            result = (
                self.weaviate_instance.client.query.get(
                    "Document",
                    ["docid", "text"],
                )
                .with_near_vector(
                    {
                        "vector": new_vector.tolist(),
                        "certainty": query.certainty,
                    },
                )
                .with_additional("distance")
                .do()
            )

            if (
                result
                and "data" in result
                and "Get" in result["data"]
                and "Document" in result["data"]["Get"]
            ):
                documents = result["data"]["Get"]["Document"]
                if documents:
                    # Find the most similar document (lowest distance)
                    most_similar_doc = min(
                        documents,
                        key=lambda doc: doc["_additional"]["distance"],
                    )
                    docid = most_similar_doc["docid"]

                    # Delete the most similar document
                    self.weaviate_instance.delete({docid})
                    logging.info(f"Deleted document with docid: {docid}")

            # Insert the new document
            new_docid = self.weaviate_instance.client.data_object.create(
                {
                    "vector": new_vector.tolist(),
                    "text": query.text,
                },
                class_name="Document",
            )
            return {
                "message": "Old document deleted and new document created successfully.",
                "docid": new_docid,
            }

        except Exception as e:
            logging.error(f"Error in update_data method: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to update data: {str(e)}",
            )
