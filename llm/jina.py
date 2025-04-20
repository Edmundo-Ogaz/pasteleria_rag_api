from langchain_community.embeddings import JinaEmbeddings
from langchain_community.document_compressors.jina_rerank import JinaRerank
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import os

# embeddings = FastEmbedEmbeddings(model_name="jinaai/jina-embeddings-v2-base-es")

embeddings = JinaEmbeddings( 
    jina_api_key=os.environ.get("JINA_API_KEY"), model_name="jina-embeddings-v2-base-es" 
)

reranker = JinaRerank(
    jina_api_key=os.environ.get("JINA_API_KEY"),
    model="jina-reranker-v1-base-en",
    top_n=3,
)

import requests

def rerank_documents(documents: list, query: str) -> dict:
    url = "https://api.jina.ai/v1/rerank"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('JINA_API_KEY')}"
    }
    data = {
        "model": "jina-reranker-v2-base-multilingual",
        "query": query,
        "top_n": 3,
        "documents": documents,
        "return_documents": False
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None
    print(response.json())
    return response.json()
