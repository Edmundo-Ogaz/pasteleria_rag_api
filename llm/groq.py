from llm.prompts import prompt_ask_product, prompt, prompt_history_retrieval, prompt_history, prompt_history_retrieval2
from llm.chroma import chroma
from repository.query import query

from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

class Groq:
    def __init__(self):
        self.__llm = ChatGroq( model="llama-3.3-70b-versatile", temperature=0.0, max_retries=2)
        rag_chain = create_stuff_documents_chain(self.__llm, prompt)
        self.__chain = create_retrieval_chain(chroma.get_retriever(), rag_chain)

        history_aware_retriever = create_history_aware_retriever(self.__llm, chroma.get_retriever(), prompt_history_retrieval2)
        rag_chain_history = create_stuff_documents_chain(self.__llm, prompt_history)
        self.chain_history = create_retrieval_chain(history_aware_retriever, rag_chain_history)

    def invoke(self, query: str) -> str:
        print("Groq invoke", query)
        result= self.__chain.invoke({"input":query})
        return result['answer']

    def is_product(self, message: str) -> bool:
        print("Groq is_product", message)
        result = False
        
        chain = prompt_ask_product | self.__llm
        response = chain.invoke({"input": message})

        if response.content.upper() != 'NO':
            result = True
        
        return result

llm = Groq()