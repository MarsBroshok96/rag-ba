import chromadb
from chromadb.utils import embedding_functions


def main():
    client = chromadb.Client(
        chromadb.config.Settings(
            persist_directory="vector_db",
            anonymized_telemetry=False,
        )
    )

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_collection(
        name="rag_pdf",
        embedding_function=embedding_function,
    )

    query = "What is machine learning?"
    results = collection.query(
        query_texts=[query],
        n_results=5,
    )

    print("\nQUERY:", query)
    print("=" * 60)

    for i, doc in enumerate(results["documents"][0]):
        print(f"\nResult {i+1}")
        print("-" * 40)
        print(doc[:500])
        print("\nMetadata:", results["metadatas"][0][i])


if __name__ == "__main__":
    main()
