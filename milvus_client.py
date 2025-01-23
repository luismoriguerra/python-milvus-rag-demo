from pymilvus import MilvusClient, model as milvus_model
from typing import List, Dict, Any
import os
from config import CLUSTER_ENDPOINT, TOKEN

# Singleton instances
_milvus_client = None
_embedding_model = None


def create_milvus_client() -> MilvusClient:
    if not CLUSTER_ENDPOINT or not TOKEN:
        raise Exception(
            "Milvus API endpoint or token not set in environment variables")

    return MilvusClient(
        uri=CLUSTER_ENDPOINT,
        token=TOKEN
    )


def get_client() -> MilvusClient:
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = create_milvus_client()
    return _milvus_client


def get_embedding_model() -> Any:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = milvus_model.DefaultEmbeddingFunction()
    return _embedding_model


def search_similar_texts(
    question: str,
    top_k: int
) -> str:
    try:
        milvus_client = get_client()
        embedding_model = get_embedding_model()
        collection_name = "my_rag_collection"

        # Optimized search parameters for better performance
        search_params = {
            "metric_type": "IP",  # Inner Product similarity metric
            "params": {
                "nprobe": 10,     # Number of clusters to search
                "ef": 64          # Higher ef value = better recall but slower search
            }
        }

        search_results = milvus_client.search(
            collection_name=collection_name,
            data=embedding_model.encode_queries([question]),
            limit=top_k,
            search_params=search_params,
            output_fields=["text"],
        )

        if not search_results or not search_results[0]:
            return "No relevant context found."

        return "\n".join([
            res["entity"]["text"] for res in search_results[0]
        ])
    except Exception as e:
        raise Exception(f"Error during Milvus search: {str(e)}")
