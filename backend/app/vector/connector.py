from backend.config import VECTOR_DB

def main():
    if VECTOR_DB == "weaviate":
        from backend.app.vector.dbs.weaviate_db import WeaviateClient
        VECTOR_DB_CLIENT = WeaviateClient()
        # You can add additional logic here
        print("WeaviateClient initialized successfully")

if __name__ == "__main__":
    main()
