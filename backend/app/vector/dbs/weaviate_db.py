import logging
from typing import List, Optional, Dict, Any, Union

import numpy as np
import weaviate
from weaviate.collections.classes.grpc import MetadataQuery
from weaviate.connect import ConnectionParams
from weaviate.exceptions import WeaviateQueryException, WeaviateConnectionError
from weaviate.util import generate_uuid5  # Generate a deterministic ID

from backend.config import (
    WEAVIATE_URL,
    WEAVIATE_SCHEMA,
    WEAVIATE_GRPC_URL,
    WEAVIATE_PREFER_GRPC
)

logger = logging.getLogger(__name__)

def check_class_exists(func):
    def wrapper(self, *args, **kwargs):
        class_name = kwargs.get('class_name', args[0] if args else None)
        if class_name and not self.has_class(class_name):
            logger.error(
                f'Class "{class_name}" not found in Weaviate. Aborting call to {func.__name__}.',
            )
            raise RuntimeError(f"Class '{class_name}' does not exist.")
        return func(self, *args, **kwargs)
    return wrapper

class WeaviateClient:
    def __init__(self):
        connection_params = ConnectionParams(
            http={
                "host": WEAVIATE_URL,  # e.g., "http://localhost"
                "port": 8080,  # Change this if you have a different port
                "secure": False  # Change to True if using HTTPS
            },
            grpc={
                "host": WEAVIATE_GRPC_URL if WEAVIATE_PREFER_GRPC else None,
                "port": 50051 if WEAVIATE_PREFER_GRPC else None,
                "secure": False if WEAVIATE_PREFER_GRPC else None
            }
        )

        # Initialize the WeaviateClient with proper connection parameters
        self.client = weaviate.WeaviateClient(
            connection_params=connection_params,
            skip_init_checks=True  # Skip health checks to avoid gRPC-related startup issues
        )
        self.offset =0
        # Attempt connection
        self.client.connect()
        logger.info("WeaviateClient connected successfully.")

        # Create schema if necessary
        self.create_schema(WEAVIATE_SCHEMA)


    def create_schema(self, schema: dict):
        try:
            if schema:
                class_name = schema.get('name')
                if class_name and not self.has_class(class_name):
                    self.client.collections.create(
                        name=schema.get("name"),
                        vectorizer_config=schema.get("vectorizer_config"),
                        generative_config=schema.get("generative_config"),
                        description = schema.get("description"),
                        properties = schema.get("properties")
                    )
                    logger.info(f"Class '{class_name}' created successfully.")
                else:
                    logger.info(f"Class '{class_name}' already exists.")
        except WeaviateConnectionError as e:
            logger.error(f"Failed to create schema due to connection issues: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during schema creation: {e}")
            raise

    def has_class(self, class_name: Optional[str] = None) -> bool:
        try:
            schema_name = self.client.collections.exists(class_name)
            return schema_name is True
        except WeaviateQueryException as e:
            logger.error(f"Failed to fetch schema for class '{class_name}': {e}")
            return False

    def delete_class(self, class_name: str):
        try:
            if self.has_class(class_name):
                self.client.collections.delete(class_name)
                logger.info(f"Class '{class_name}' deleted successfully.")
            else:
                logger.warning(f"Class '{class_name}' does not exist.")
        except WeaviateConnectionError as e:
            logger.error(f"Failed to delete class '{class_name}' due to connection issues: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while deleting class '{class_name}': {e}")
            raise

    def insert_objects(self, class_name: str, objects: List[Dict[str, Any]],
                       vectors: Optional[List[List[float]]] = None):
        try:
            collection = self.client.collections.get(class_name)
            with collection.batch.dynamic() as batch:
                for i, obj in enumerate(objects):
                    obj['doc_id'] = self.offset + i + 1
                    obj_uuid = generate_uuid5(obj['doc_id'])
                    vector = vectors[i] if vectors else None
                    batch.add_object(properties=obj, uuid=obj_uuid, vector=vector)

            logger.info(f"Inserted {len(objects)} objects into class '{class_name}'.")
        except WeaviateConnectionError as e:
            logger.error(f"Failed to insert objects into class '{class_name}' due to connection issues: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while inserting objects into class '{class_name}': {e}")
            raise

    @check_class_exists
    def search(self, class_name: str, vector: Union[List[float], dict], limit: int = 7) -> List[Dict[str, Any]]:
        try:
            # Nếu vector là một danh sách (list), chuyển đổi nó thành dạng dictionary cần thiết
            if isinstance(vector, list):
                vector = {"average_embedding": vector}  # Chuyển thành dict với key là 'average_embedding'

            if not isinstance(vector, dict):
                raise ValueError("Vector phải là một dict hoặc một list.")

            if 'average_embedding' not in vector:
                raise ValueError("'average_embedding' không tồn tại trong vector.")

            collection = self.client.collections.get(class_name)
            response = collection.query.near_vector(
                near_vector=vector['average_embedding'],
                limit=limit,
                return_metadata=MetadataQuery(distance=True)
            )

            return response  # Trả về kết quả truy vấn
        except Exception as e:
            print(f"Đã xảy ra lỗi: {e}")
            return []