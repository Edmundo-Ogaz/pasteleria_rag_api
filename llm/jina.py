from langchain_community.embeddings import JinaEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import os

# embeddings = FastEmbedEmbeddings(model_name="jinaai/jina-embeddings-v2-base-es")

embeddings = JinaEmbeddings( 
    jina_api_key=os.environ.get("JINA_API_KEY"), model_name="jina-embeddings-v2-base-es" 
)