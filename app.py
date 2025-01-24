from flask import Flask, request, Response

from langchain.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

from langchain_groq import ChatGroq

from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import SQLAlchemyError

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.messages import HumanMessage, AIMessage

from langchain_core.runnables import ConfigurableFieldSpec

import os
from dotenv import load_dotenv
load_dotenv()

HISTORY_LENGTH = 2
DATABASE_URL = "sqlite:///db/chroma.sqlite3"
Base = declarative_base()

class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, nullable=False)
    messages = relationship("Message", back_populates="session")
    product = relationship("Product", back_populates="session")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    session = relationship("Session", back_populates="messages")

class Product(Base):
    __tablename__ = "product"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    name = Column(Text, nullable=False)
    session = relationship("Session", back_populates="product")

# Create the database and the tables
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_message(session_id: str, role: str, content: str):
    db = next(get_db())
    try:
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if not session:
            session = Session(session_id=session_id)
            db.add(session)
            db.commit()
            db.refresh(session)

        db.add(Message(session_id=session.id, role=role, content=content))
        db.commit()
    except SQLAlchemyError as e:
        print(e)
        db.rollback()
    finally:
        db.close()

def save_product(session_id: str, name: str):
    db = next(get_db())
    try:
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if not session:
            session = Session(session_id=session_id)
            db.add(session)
            db.commit()
            db.refresh(session)

        db.add(Product(session_id=session.id, name=name))
        db.commit()
    except SQLAlchemyError as e:
        print(e)
        db.rollback()
    finally:
        db.close()

def load_session_history(session_id: str) -> BaseChatMessageHistory:
    print("load_session_history")
    db = next(get_db())
    chat_history = ChatMessageHistory()
    try:
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if session:
            messages = session.messages[-(HISTORY_LENGTH):]
            for message in messages:
                chat_history.add_message({"role": message.role, "content": message.content})
    except SQLAlchemyError as e:
        print(e)
        pass
    finally:
        db.close()

    return chat_history

def load_product(session_id: str) -> str:
    print("load_product")
    db = next(get_db())
    product = ''
    try:
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if session:
            products = session.product
            for element in products:
                product = element
    except SQLAlchemyError as e:
        print(e)
        return ''
        pass
    finally:
        db.close()

    return product

embeddings = FastEmbedEmbeddings(model_name="jinaai/jina-embeddings-v2-base-es")

# - Para preguntas sin información en el contexto, responde: 'Lo siento, no puedo responder esta pregunta con la información disponible. ¿Hay algo más en lo que pueda ayudarte?'
prompt_history = ChatPromptTemplate.from_messages([
    ("system", """
        Eres un asistente amigable de la Pastelería la Palmera. Tu rol es proporcionar información precisa sobre nuestros productos y servicios.

        Directrices principales:
        - Responde de forma precisa, concisa y breve
        - Utiliza EXCLUSIVAMENTE la información del contexto proporcionado
        - No inferir ni inventar información adicional
        - No realices suposiciones sobre productos, precios o servicios no mencionados explícitamente
        - Responde siempre en el mismo idioma de la pregunta
        - Mantén un tono amable y servicial

        Contexto:
        {context}
    """
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# - Para preguntas sin información en el contexto, responde: 'Lo siento, no puedo responder esta pregunta con la información disponible. ¿Hay algo más en lo que pueda ayudarte?'
prompt = ChatPromptTemplate.from_messages([
    ("system", """
        Eres un asistente amigable de la Pastelería la Palmera. Tu rol es proporcionar información precisa sobre nuestros productos y servicios.

        Directrices principales:
        - Responde de forma precisa, concisa y breve
        - Utiliza EXCLUSIVAMENTE la información del contexto proporcionado
        - No inferir ni inventar información adicional
        - No realices suposiciones sobre productos, precios o servicios no mencionados explícitamente
        - Responde siempre en el mismo idioma de la pregunta
        - Mantén un tono amable y servicial

        Contexto:
        {context}
    """
    ),
    ("human", "{input}"),
])

prompt_history_retrieval = ChatPromptTemplate.from_messages(
    [
        ("system", """
            Dado un historial de chat y la ultima pregunta del usuario que podría hacer referencia al contexto del historial del chat, 
            formula un pregunta tomando en cuenta el ultimo mensaje del usuario (puedes apoyarte de los mensaje anteriores si la ultima pregunta hace referencia a estos).
            No respondas la pregunta, simplemente reformula si es necesario y de lo contrario devuelvela como está.
        """),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

prompt_ask_product = PromptTemplate.from_template("""
    <s>[INST] Analiza el siguiente mensaje y extrae nombres exactos de productos de la Pastelería La Palmera según estas reglas:
    1. PRODUCTOS VÁLIDOS (ignorar variaciones en mayúsculas/minúsculas):
    - Tortas (incluye: Torta de Chocolate, Torta Africana, etc.)
    - Productos de cotelería 
    - Kutchen (incluye: Kutchen de Frambuesa, etc.)
    - Cheesecake (incluye: Cheesecake de Frutilla, etc.)
    - Pie (incluye: Pie de Limón, etc.)
    - Tartaleta (incluye: Tartaleta de Frutas, etc.)
    2. FORMATO DE RESPUESTA:
    - Si encuentra un producto válido: Devolver el nombre EXACTO del producto en mayúsculas
    - Si encuentra múltiples productos: Devolver cada producto en una nueva línea
    - Si no encuentra productos válidos o hay ambigüedad: Devolver 'NO'
    3. REGLAS DE EXTRACCIÓN:
    - Ignorar palabras como "quiero", "necesito", "tienes", etc.
    - Considerar válidas las variantes específicas de los productos base
    - No hacer inferencias ni suposiciones sobre productos no mencionados explícitamente
    - No incluir explicaciones adicionales [/INST] </s>
    
    [INST] Mensaje a analizar: {input}
    Respuesta: [/INST]
"""
)

def is_product(message) -> bool:
    print("is_product")
    result = False
    
    chain = prompt_ask_product | llm
    response = chain.invoke({"input": message})

    print("is_product", response)
    print("is_product", response.content.upper())
    if response.content.upper() != 'NO':
        print("is not no")
        result = True
    else:
        print("is no")
    
    return result

def get_session_history(session_id: str, message: str) -> BaseChatMessageHistory:
    print("get_session_chat_history session_id", session_id, message)

    chat_history = ChatMessageHistory()
        
    if is_product(message):
        session_product[session_id] = message
        chat_history.add_message({"role": "human", "content": message})
        save_product(session_id, message)
    else:
        if session_id in session_product:
            text = session_product[session_id]
            chat_history.add_message({"role": "human", "content": text})
        else:
            product = load_product(session_id)
            if product:
                session_product[session_id] = product

    if session_id not in store:
        print("get_session_chat_history")
        store[session_id] = load_session_history(session_id)
    else:  
        session = store[session_id]
        messages = session.messages[-(HISTORY_LENGTH):]
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "human"
                content = message.content
            elif isinstance(message, AIMessage):
                role = "ai"
                content = message.content
            else:
                role = message['role']
                content = message['content']
            chat_history.add_message({"role": role, "content": content})
        store[session_id] = chat_history

    print("get_session_history response", store)
    return store[session_id]

vectordb= Chroma(
    persist_directory='db', embedding_function=embeddings, collection_name="mi_coleccion",
)

retriever = vectordb.as_retriever(
    search_type= "similarity_score_threshold",
    search_kwargs={
        "k":3,
        "score_threshold":0.0
    }
)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    max_retries=2,
)

rag_chain = create_stuff_documents_chain(llm, prompt)

chain = create_retrieval_chain(retriever, rag_chain)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, prompt_history_retrieval
)

rag_chain_history = create_stuff_documents_chain(llm, prompt_history)

chain_history = create_retrieval_chain(history_aware_retriever, rag_chain_history)

store = {}
session_product = {}

conversational_rag_chain = RunnableWithMessageHistory(
    chain_history,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="Unique identifier for the session.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="message",
            annotation=str,
            name="Message",
            description="Message.",
            default="",
            is_shared=True,
        ),
    ],
)

def invoke_and_save(session_id, input_text):
    # Save the user question with role "human"
    save_message(session_id, "human", input_text)
    
    result = conversational_rag_chain.invoke(
        {"input": input_text},
        config={"configurable": {"session_id": session_id, "message": input_text}}
    )

    # Save the AI answer with role "ai"
    save_message(session_id, "ai", result["answer"])
    return result

app = Flask(__name__)

@app.route("/health", methods=["GET"]) 
def health():
    message = "OK"
    response = Response(message, status=200)
    return response

@app.route("/ask", methods=["POST"]) 
def ask():
    json_content = request.json
    query = str(json_content.get("query")).lower()
    session_id = request.headers.get("sessionId")
    print(f"query: {query}")
    print(f"sessionId: {session_id}")

    vectordb= Chroma(
        persist_directory='db', embedding_function=embeddings, collection_name="mi_coleccion",
    )

    response = ''

    result = vectordb.similarity_search_with_relevance_scores( query, k=3, score_threshold=0.0 )
    for doc, i in result:
        print(doc.page_content, i)
        response += f'{doc.page_content}\n'
        print("---")

    return {"respuesta": response}

@app.route("/ask-model", methods=["POST"]) 
def ask_model():
    json_content = request.json
    query = str(json_content.get("query")).lower()
    session_id = request.headers.get("sessionId")
    print(f"query: {query}")
    print(f"session_id: {session_id}")

    result= chain.invoke({"input":query})
    print(result)
    response_answer= {"respuesta": result['answer']}
    return response_answer

@app.route("/ask-model-history", methods=["POST"]) 
def ask_model_history():
    json_content = request.json
    query = str(json_content.get("query")).lower()
    session_id = request.headers.get("sessionId")
    print(f"query: {query}")
    print(f"session_id: {session_id}")

    result = invoke_and_save(session_id, query)
    print(result)
    print("----------------")
    print(store)
    print("----------------")
    print(session_product)
    response_answer= {"respuesta": result['answer']}
    return response_answer
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host='0.0.0.0', port=port,debug=True)