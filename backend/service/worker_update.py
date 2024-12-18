from __future__ import annotations

import logging

from rapidfuzz.distance import Levenshtein
from txtai import Embeddings

from backend.app.vector.dbs.weaviate_db import WeaviateClient


def get_similarity(old_content, new_content):
    distance = Levenshtein.distance(old_content, new_content)
    similarity = 1 - (distance / max(len(new_content), len(old_content)))
    return similarity


class WorkerUpdate:
    def __init__(self, weaviate_instance: WeaviateClient) -> None:
        self.weaviate_instance = weaviate_instance
        self.low_similarity_laws: list[
            str
        ] = []  # To store laws with low similarity
        self.embeddings = Embeddings(
            {"path": "sentence-transformers/all-MiniLM-L6-v2"},
        )

    def get_existing_laws(self):
        existing_laws = {}
        try:
            results = self.weaviate_instance.client.query.get(
                "Document",
                ["law_number", "text"],
            ).do()
            documents = (
                results.get("data", {})
                .get(
                    "Get",
                    {},
                )
                .get("Document", [])
            )
            for doc in documents:
                law_number = doc.get("law_number")
                text = doc.get("text")
                if law_number and text:
                    existing_laws[law_number] = text

        except Exception as e:
            logging.error(
                f"Error while retrieving existing laws from Weaviate: {e}",
            )
            raise RuntimeError(f"Failed to retrieve existing laws: {e}")

        return existing_laws

    def update_laws(self, new_laws: dict):
        """
        Update Weaviate with the new set of laws by comparing with the existing data.
        :param new_laws: Dictionary of new laws where key is law number and value is content.
        """
        try:
            # Retrieve existing laws from Weaviate
            existing_laws = self.get_existing_laws()

            # Compare new laws with existing ones
            for law_number, new_content in new_laws.items():
                # Check if law already exists in Weaviate
                if law_number in existing_laws:
                    old_content = existing_laws[law_number]
                    similarity = get_similarity(old_content, new_content)

                    # If similarity is low, mark for re-embedding
                    if similarity <= 0.85:
                        self.low_similarity_laws.append(law_number)

                else:
                    # New law, embed and save
                    self.embed_and_save(law_number, new_content)

            # Delete old versions of laws with low similarity
            self.delete_low_similarity_laws()

            # Re-embed and save laws that have low similarity
            for law_number in self.low_similarity_laws:
                self.embed_and_save(law_number, new_laws[law_number])

        except Exception as e:
            logging.error(f"Error while updating laws: {e}")
            raise RuntimeError(f"Failed to update laws: {e}")

    def embed_and_save(self, law_number, content):
        """
        Embed the text and save the embedding data into Weaviate.
        :param law_number: The law number to embed and save.
        :param content: The content of the law to embed and save.
        """
        try:
            if not content.strip():
                raise ValueError("No content provided for embedding")

            # Generate embedding using txtai
            embedding = self.embeddings.transform(content)

            # Prepare data for saving
            embedding_data = {
                "law_number": law_number,
                "embedding": embedding,
                "text": content.strip(),
            }

            # Save data to Weaviate
            self.weaviate_instance.append(
                [
                    (
                        embedding_data["law_number"],
                        embedding_data["embedding"],
                        embedding_data["text"],
                    ),
                ],
            )

            logging.info(
                f"Successfully embedded and saved law {law_number} into Weaviate.",
            )

        except Exception as e:
            logging.error(
                f"Error while embedding and saving law {law_number}: {e}",
            )
            raise RuntimeError(
                f"Failed to embed and save law {law_number}: {e}",
            )

    def delete_low_similarity_laws(self):
        """
        Handle laws with low similarity by deleting them from Weaviate.
        """
        try:
            for law_number in self.low_similarity_laws:
                self.weaviate_instance.delete_law_number(law_number)

        except Exception as e:
            logging.error(
                f"Error while deleting laws with low similarity: {e}",
            )
            raise RuntimeError(
                f"Failed to delete laws with low similarity: {e}",
            )
