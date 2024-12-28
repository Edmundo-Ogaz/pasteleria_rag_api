from flask import Flask, request

from langchain.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# from langchain_community.llms import Ollama
# from langchain_groq import ChatGroq

embeddings = FastEmbedEmbeddings(model_name="jinaai/jina-embeddings-v2-base-es")
vectordb = Chroma( persist_directory='db', embedding_function=embeddings, collection_name="mi_coleccion", )
# llm= Ollama(model="llama3.1:8b", temperature=0.3, num_predict=512)
raw_prompt= PromptTemplate.from_template("""
    <s>[INST] Eres un asistente de la Pastelería la Palmera, que responde únicamente con la información proporcionada. Si no sabes la respuesta con la información proporcionada. Si no saber la respuesta con la información proporcionada, sé honesto y responde: 'Lo siento, no sé la respuesta a esta pregunta' [/INST] </s>
    [INST] {input}
            Context: {context}
            Answer: 
    [/INST]
"""
)

app = Flask(__name__)

@app.route("/ask", methods=["POST"]) 
def ask():
    json_content = request.json
    query= json_content.get("query")
    print(f"query: {query}")

    result = vectordb.similarity_search(query, k=1)
    print("\nResultados de la búsqueda:")
    for doc in result:
        print(doc.page_content)
        print("---")
    
    response_answer= {"respuesta": result[0].page_content.strip()}
    return response_answer

@app.route("/ask-model", methods=["POST"]) 
def ask_model():
    json_content = request.json
    query= json_content.get("query")
    print(f"query: {query}")

    retriever = vectordb.as_retriever(
        search_type= "similarity_score_threshold",
        search_kwargs={
            "k":3,
            "score_threshold":0.1
        }
    )

    llm = null
    # llm = ChatOpenAI(model="gpt-3.5-turbo")
    # llama3-groq-70b-8192-tool-use-preview
    # mixtral-8x7b-32768
    # llm = ChatGroq(
    #     model="llama3-groq-70b-8192-tool-use-preview",
    #     temperature=0.0,
    #     max_retries=2,
    #     # other params...
    # )
    document_chain = create_stuff_documents_chain( llm, raw_prompt )
    chain = create_retrieval_chain(retriever, document_chain)
    result = chain.invoke({"input":query})
    print(result)
    response_answer= {"respuesta": result['answer'].replace("<s>[INST]", "").replace("[/INST]</s>","").strip()}
    return response_answer
    

if __name__ == "__main__":
    app.run(debug=True)