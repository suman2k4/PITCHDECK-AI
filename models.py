from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    sessions = relationship('Session', back_populates='user')

class Session(Base):
    __tablename__ = 'sessions'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    persona = Column(String(50))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    user = relationship('User', back_populates='sessions')
    files = relationship('PitchFile', back_populates='session')
    feedback = relationship('Feedback', back_populates='session', uselist=False)
    answers = relationship('Answer', back_populates='session')
class Answer(Base):
    __tablename__ = 'answers'
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('sessions.id'))
    question = Column(Text)
    answer = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    session = relationship('Session', back_populates='answers')

class PitchFile(Base):
    __tablename__ = 'pitch_files'
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('sessions.id'))
    filename = Column(String(200))
    filetype = Column(String(10))
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)
    session = relationship('Session', back_populates='files')

class Feedback(Base):
    __tablename__ = 'feedback'
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('sessions.id'))
    clarity_score = Column(String(10))
    realism_score = Column(String(10))
    relevance_score = Column(String(10))
    actionability_score = Column(String(10))
    improvement = Column(Text)
    recommendations = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    session = relationship('Session', back_populates='feedback')

class CoachingFeedback(Base):
    __tablename__ = 'coaching_feedback'
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer)
    question = Column(Text)
    answer = Column(Text)
    score = Column(String(10))
    notes = Column(Text)
    suggestion = Column(Text)
    slide_refs = Column(Text)  # JSON-encoded list of slide numbers
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
