from langchain.vectorstores import Chroma
from llm.jina import embeddings

class ChromaDB:
    def __init__(self):
        self.__db = Chroma(
            persist_directory='db', embedding_function=embeddings, collection_name="mi_coleccion",
        )

    def get_similarity_whith_scores(self, query: str):
        return self.__db.similarity_search_with_relevance_scores( query, k=3, score_threshold=0.0 )

    def get_similarity(self, query: str):
        response = ''

        result = self.__db.similarity_search_with_relevance_scores( query, k=3, score_threshold=0.0 )
        for doc, i in result:
            print(doc.page_content, i)
            response += f'{doc.page_content}\n'
            print("---")

        return response
    
    def get_retriever(self):
        retriever = self.__db.as_retriever(
            search_type= "similarity_score_threshold",
            search_kwargs={
                "k":3,
                "score_threshold":0.0
            }
        )

        return retriever

chroma = ChromaDB()