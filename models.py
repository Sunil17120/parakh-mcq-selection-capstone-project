from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Float, DateTime
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    tenth_percentage = Column(Float)
    twelfth_percentage = Column(Float)
    ability_score = Column(Float, default=0.0)
    correct_mcq_count = Column(Integer, default=0)
    
    # This is the field that was causing the error
    is_admin = Column(Boolean, default=False) 

class Question(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    options = Column(String) # Stored as comma-separated string
    answer = Column(String)
    difficulty = Column(Float)
    type = Column(String) # 'mcq' or 'scenario'
    
    # IRT Parameters
    irt_difficulty = Column(Float, default=0.0)
    irt_discrimination = Column(Float, default=1.0)

class StudentPerformance(Base):
    __tablename__ = "student_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    question_id = Column(Integer, ForeignKey("questions.id"))
    
    correct_attempts = Column(Integer, default=0)
    total_attempts = Column(Integer, default=0)
    score = Column(Float, nullable=True) # For scenario questions
    last_attempt_date = Column(DateTime, default=datetime.utcnow)