from __future__ import annotations

import logging

import weaviate
from txtai.ann import ANN
from weaviate import Client
from weaviate.util import generate_uuid5

# Configure logger
logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

DEFAULT_SCHEMA = {
    "class": "Document",
    "properties": [
        {"name": "docid", "dataType": ["int"]},
        {"name": "text", "dataType": ["text"]},
        {"name": "law_number", "dataType": ["int"]},
    ],
    "vectorIndexConfig": {"distance": "cosine"},
}

DEFAULT_BATCH_CONFIG = {
    "batch_size": None,
    "creation_time": None,
    "timeout_retries": 3,
    "connection_error_retries": 3,
    "weaviate_error_retries": None,
    "callback": None,
    "dynamic": False,
    "num_workers": 1,
}


def normalize_cosine_distance(cosine_distance):
    # Normalize Weaviate's cosine distance to a similarity score
    return 1 - cosine_distance


def check_index_exists(func):
    """Decorator to ensure the index exists before executing the method."""

    def wrapper(self, *args, **kwargs):
        if not self._index_exists():
            logger.error(
                f'Index "{self.index_name}" not found in Weaviate. Aborting call to {func.__name__}.',
            )
            raise RuntimeError(f"Index '{self.index_name}' does not exist.")
        return func(self, *args, **kwargs)

    return wrapper


def _is_valid_schema(schema):
    """Validate schema against required properties and vector config."""
    required_properties = [
        {"name": "docid", "dataType": ["int"]},
        {"name": "text", "dataType": ["text"]},
        {"name": "law_number", "dataType": ["int"]},
    ]
    distance_metric = schema.get(
        "vectorIndexConfig",
        {},
    ).get("distance", "cosine")

    if distance_metric != "cosine":
        return False

    for prop in required_properties:
        if not any(
            prop["name"] == required["name"]
            and prop["dataType"] == required["dataType"]
            for required in schema["properties"]
        ):
            return False
    return True


class Weaviate(ANN):
    """
    Builds an ANN index using the Weaviate vector search engine.
    Implements singleton behavior to avoid redundant initialization.
    """

    _instance = None  # Class-level instance for singleton behavior

    def __new__(cls, config):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, config):
        if self.initialized:
            return

        super().__init__(config)
        self.weaviate_config = config.get("weaviate", {})
        url = self.weaviate_config.get("url", "http://localhost:8080")
        self.client = Client(url)

        if not self.client.is_ready():
            raise RuntimeError(
                "Weaviate server is not ready. Ensure the server is running.",
            )

        self.index_name = None
        self.config["offset"] = self.config.get("offset", 0)
        self.overwrite_index = self.weaviate_config.get(
            "overwrite_index",
            False,
        )

        # Create schema only if necessary
        self._create_schema()

        # Configure batch settings for Weaviate client
        batch_config = self.weaviate_config.get("batch", DEFAULT_BATCH_CONFIG)
        self._configure_client(**batch_config)

        self.initialized = True
        logger.info("Weaviate instance initialized successfully.")

    @classmethod
    def get_instance(cls):
        if cls._instance is None or not cls._instance.initialized:
            raise RuntimeError(
                "Weaviate not created. You can initialize the Weaviate instance.",
            )
        return cls._instance

    def _configure_client(self, **batch_config):
        """Configures Weaviate batch client."""
        self.client.batch.configure(**batch_config)

    def _create_schema(self):
        """Create the schema if it doesn't exist or needs to be overwritten."""
        schema = self.weaviate_config.get("schema", DEFAULT_SCHEMA)

        if not _is_valid_schema(schema):
            raise weaviate.exceptions.SchemaValidationException(
                f"Class {schema['class']} must have properties named 'docid', 'text', and 'law_number'.",
            )

        try:
            if self.client.schema.contains(schema):
                if not self.overwrite_index:
                    logger.info(
                        f"Schema {schema['class']} already exists. Skipping creation.",
                    )
                    self.index_name = schema["class"]
                    return
                else:
                    logger.warning(f"Overwriting schema {schema['class']}.")
                    self.client.schema.delete_class(schema["class"])

            # Create a new schema
            self.create_class(schema)

        except weaviate.exceptions.UnexpectedStatusCodeException as e:
            logger.error(
                f"Unexpected error occurred while checking or creating schema: {e}",
            )
            raise

        self.index_name = schema["class"]

    def create_class(self, schema):
        """Create a schema class in Weaviate."""
        try:
            self.client.schema.create_class(schema)
            logger.info(f"Schema {schema['class']} created successfully.")
        except weaviate.exceptions.SchemaValidationException as e:
            logger.error(f"Schema validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            raise

    def _index_exists(self):
        """Check if the index exists in Weaviate schema."""
        try:
            self.client.schema.get(self.index_name)
            return True
        except weaviate.exceptions.UnexpectedStatusCodeException:
            return False

    def index(self, embeddings):
        """Index embeddings and their associated content."""
        self.append(embeddings)

    def append(self, embeddings):
        """Append embeddings and content to the index."""
        if not embeddings:
            logger.warning("No embeddings provided to append.")
            return

        with self.client.batch as batch:
            for law_number, embedding, text in embeddings:
                random_identifier = str(self.config["offset"])
                object_uuid = generate_uuid5(random_identifier)

                try:
                    batch.add_data_object(
                        data_object={
                            "docid": self.config["offset"],
                            "text": text,
                            "law_number": law_number,
                        },
                        class_name=self.index_name,
                        vector=embedding,
                        uuid=object_uuid,
                    )
                    self.config["offset"] += 1
                except Exception as e:
                    logger.error(f"Failed to add object to Weaviate: {e}")
                    continue

    def _get_uuid_from_docid(self, docid):
        """Retrieve UUID from docid."""
        results = (
            self.client.query.get(self.index_name)
            .with_additional("id")
            .with_where(
                {
                    "path": ["docid"],
                    "operator": "Equal",
                    "valueInt": docid,
                },
            )
            .do()
        )
        if not results["data"]["Get"][self.index_name]:
            raise RuntimeError(f"No document found with docid: {docid}")
        return results["data"]["Get"][self.index_name][0]["_additional"]["id"]

    @check_index_exists
    def delete(self, ids):
        """Delete objects by their docid."""
        for docid in ids:
            try:
                uuid = self._get_uuid_from_docid(docid)
                self.client.data_object.delete(
                    uuid,
                    class_name=self.index_name,
                )
                logger.info(f"Deleted document with docid {docid}.")
            except Exception as e:
                logger.error(
                    f"Failed to delete document with docid {docid}: {e}",
                )

    @check_index_exists
    def delete_law_number(self, law_number):
        """
        Delete object by its law number.
        :param law_number: Law number to be deleted.
        """
        try:
            # Query to find the object by law_number
            results = (
                self.client.query.get(self.index_name, ["_additional { id }"])
                .with_where(
                    {
                        "path": ["law_number"],
                        "operator": "Equal",
                        "valueInt": law_number,
                    },
                )
                .do()
            )

            documents = (
                results.get("data", {})
                .get(
                    "Get",
                    {},
                )
                .get(self.index_name, [])
            )
            if documents:
                for document in documents:
                    object_id = document.get("_additional", {}).get("id")
                    if object_id:
                        self.client.data_object.delete(
                            object_id,
                            class_name=self.index_name,
                        )
                        logging.info(
                            f"Deleted document with law number {law_number}.",
                        )
                    else:
                        logging.warning(
                            f"No object ID found for law number {law_number}.",
                        )
            else:
                logging.warning(
                    f"No document found with law number {law_number}.",
                )
        except Exception as e:
            logging.error(
                f"Failed to delete document with law number {law_number}: {e}",
            )

    @check_index_exists
    def search(self, queries, limit):
        """Search for nearest neighbors using embeddings."""
        near_vector = {"vector": queries[0]}
        results = (
            self.client.query.get(
                self.index_name,
                properties=[
                    "docid",
                    "text",
                    "law_number",
                ],
            )
            .with_additional("distance")
            .with_near_vector(near_vector)
            .with_limit(limit)
            .do()
        )

        return [
            {
                "docid": result["docid"],
                "text": result["text"],
                "law_number": result["law_number"],
                "score": normalize_cosine_distance(
                    result["_additional"]["distance"],
                ),
            }
            for result in results["data"]["Get"][self.index_name]
        ]

    @check_index_exists
    def count(self):
        """Count the number of objects in the index."""
        results = (
            self.client.query.aggregate(
                self.index_name,
            )
            .with_meta_count()
            .do()
        )
        return results["data"]["Aggregate"][self.index_name][0]["meta"][
            "count"
        ]

    def save(self, path):
        """Save method stub."""
        logger.warning(
            "The save method has no effect on the embeddings stored in Weaviate. "
            "Use Weaviate's Backup API for this functionality.",
        )

    def load(self, path):
        """Load method stub."""
        logger.warning(
            "The load method has no effect on the embeddings stored in Weaviate. "
            "Use Weaviate's Backup API for this functionality.",
        )
