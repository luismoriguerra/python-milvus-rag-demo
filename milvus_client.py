from pymilvus import MilvusClient, model as milvus_model
from typing import List, Dict, Any
import os
from config import CLUSTER_ENDPOINT, TOKEN

# Singleton client instance
_milvus_client = None


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
    return milvus_model.DefaultEmbeddingFunction()


def search_similar_texts(
    question: str,
    top_k: int
) -> str:

    milvus_client = get_client()

    embedding_model = get_embedding_model()

    collection_name = "my_rag_collection"

    search_results = milvus_client.search(
        collection_name=collection_name,
        data=embedding_model.encode_queries([question]),
        limit=top_k,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["text"],
    )

    return "\n".join([
        res["entity"]["text"] for res in search_results[0]
    ])
