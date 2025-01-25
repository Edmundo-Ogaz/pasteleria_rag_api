from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.messages import HumanMessage, AIMessage
import os

from llm.groq import llm
from repository.query import query

class Session:
    def __init__(self):
        self.__history_messages = {}
        self.__history_products = {}
        self.__query = query

        self.__conversational_rag_chain = RunnableWithMessageHistory(
            llm.chain_history,
            self.__get_session_history,
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
        
    def __get_session_history(self, session_id: str, message: str) -> BaseChatMessageHistory:
        print("get_session_chat_history session_id", session_id, message)

        chat_history = ChatMessageHistory()
            
        if llm.is_product(message):
            self.__history_products[session_id] = message
            chat_history.add_message({"role": "human", "content": message})
            self.__query.save_product(session_id, message)
        else:
            if session_id in self.__history_products:
                text = self.__history_products[session_id]
                chat_history.add_message({"role": "human", "content": text})
            else:
                product = self.__query.load_product(session_id)
                if product:
                    self.__history_products[session_id] = product

        if session_id not in self.__history_messages:
            print("get_session_chat_history")
            self.__history_messages[session_id] = self.__query.load_session_history(session_id)
        else:  
            session = self.__history_messages[session_id]
            messages = session.messages[-(int(os.environ.get("HISTORY_LENGTH"))):]
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
            self.__history_messages[session_id] = chat_history

        print("get_session_history response", self.__history_messages)
        return self.__history_messages[session_id]
    
    def invoke_with_history(self, session_id, input_text) -> str:
        print("Groq invoke_with_history", session_id, input_text)

        self.__query.save_message(session_id, "human", input_text)
        
        result = self.__conversational_rag_chain.invoke(
            {"input": input_text},
            config={"configurable": {"session_id": session_id, "message": input_text}}
        )
        response = result["answer"]

        self.__query.save_message(session_id, "ai", response)

        return response
    
session = Session()