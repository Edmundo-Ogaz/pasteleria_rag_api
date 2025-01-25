from sqlalchemy.exc import SQLAlchemyError
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import os

from repository.model import Session, Message, Product, get_db

class Query:

    def save_message(self, session_id: str, role: str, content: str):
        print("save_message request", session_id, role, content)
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

    def save_product(self, session_id: str, name: str):
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

    def load_session_history(self, session_id: str) -> BaseChatMessageHistory:
        print("load_session_history request", session_id)
        db = next(get_db())
        chat_history = ChatMessageHistory()
        try:
            session = db.query(Session).filter(Session.session_id == session_id).first()
            if session:
                messages = session.messages[-(int(os.environ.get("HISTORY_LENGTH"))):]
                for message in messages:
                    chat_history.add_message({"role": message.role, "content": message.content})
            print("load_session_history reponse", chat_history)
        except SQLAlchemyError as e:
            print("load_session_history error", e)
            pass
        except Exception as e:
            print("load_session_history exception", e)
        finally:
            db.close()

        return chat_history

    def load_product(self, session_id: str) -> str:
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
    
query = Query()