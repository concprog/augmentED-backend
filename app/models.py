from sqlalchemy import Column
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import Integer, String, ForeignKey, JSON, Float, LargeBinary

from app.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)


class Books(Base):
    __tablename__ = "books_store"

    id = Column(Integer, primary_key=True, index=True)
    subject = Column(String)
    document = Column(LargeBinary(length = 1048576), nullable=True)


class PastYearQuestions(Base):
    __tablename__ = "past_year"

    id = Column(Integer, primary_key=True, index=True)
    test_id = Column(Integer)
    test_type = Column(String)
    test_year = Column(Integer)
    question_text = Column(String)
    
