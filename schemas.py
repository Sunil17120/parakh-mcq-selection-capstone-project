from pydantic import BaseModel, Field
from typing import Optional, List

# Pydantic models for data validation and serialization

# --- User Schemas ---
class UserBase(BaseModel):
    email: str
    full_name: str
    tenth_percentage: int
    twelfth_percentage: int

class UserCreate(UserBase):
    password: str

class UserOut(UserBase):
    id: int
    ability_score: float

    class Config:
        orm_mode = True

# --- Question Schemas ---
class QuestionBase(BaseModel):
    id: int
    text: str
    type: str
    options: Optional[List[str]] = None
    difficulty: float
    irt_difficulty: float
    irt_discrimination: float

    class Config:
        orm_mode = True

class Question(QuestionBase):
    answer: str

    class Config:
        orm_mode = True

# --- Answer Schemas ---
class AnswerSubmit(BaseModel):
    answer: str

class AnswerResult(BaseModel):
    is_correct: Optional[bool] = None
    correct_answer: Optional[str] = None
    score: Optional[float] = None

# --- Token Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
class PerformanceHistory(BaseModel):
    question_text: str
    is_correct: Optional[bool]
    score: Optional[float]
    date: str

class UserProgress(BaseModel):
    total_questions_attempted: int
    total_correct_mcq: int
    scenario_average_score: float
    current_ability_score: float
    accuracy_rate: float
    recent_history: List[PerformanceHistory]
class UserOut(UserBase):
    id: int
    ability_score: float
    is_admin: bool = False # <--- Add this line

    class Config:
        orm_mode = True
class StudentPerformanceSummary(BaseModel):
    full_name: str
    email: str
    questions_attempted: int
    accuracy: float
    ability_score: float

class AdminDashboardStats(BaseModel):
    total_students: int
    class_average_accuracy: float
    student_list: List[StudentPerformanceSummary]