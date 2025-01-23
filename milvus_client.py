from pymilvus import MilvusClient, model as milvus_model
from typing import List, Dict, Any
import os

def create_milvus_client() -> MilvusClient:
    CLUSTER_ENDPOINT = os.getenv("API")
    TOKEN = os.getenv("TOKEN")

    if not CLUSTER_ENDPOINT or not TOKEN:
        raise Exception("Milvus API endpoint or token not set in environment variables")

    return MilvusClient(
        uri=CLUSTER_ENDPOINT,
        token=TOKEN
    )

def get_embedding_model() -> Any:
    return milvus_model.DefaultEmbeddingFunction()

def search_similar_texts(
    client: MilvusClient,
    collection_name: str,
    question: str,
    top_k: int,
    embedding_model: Any
) -> str:
    search_results = client.search(
        collection_name=collection_name,
        data=embedding_model.encode_queries([question]),
        limit=top_k,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["text"],
    )

    return "\n".join([
        res["entity"]["text"] for res in search_results[0]
    ]) 