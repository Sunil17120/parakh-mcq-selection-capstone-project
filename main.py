# main.py (Final Confirmed Correct Version with NLP Integration)

from datetime import timedelta 
import os
import sys
import warnings
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

# Database and Security Imports
import crud, models, schemas, security 
from database import engine, get_db 
from models import User 
from schemas import AnswerSubmit, UserOut, QuestionBase, AnswerResult, AdminDashboardStats 

# ML/AI Imports
from ml_predictor import predict_multi_class_type 
from gen_ai_service import generate_questions_with_ai
import ml_service 
import nlp_service 
from dotenv import load_dotenv

# Scheduler Imports
from apscheduler.schedulers.background import BackgroundScheduler
from scheduler_job import scheduled_training_job


# Load environment variables from .env file
load_dotenv()
# --------------------------------------

# Suppress Pydantic V2 warnings
warnings.filterwarnings("ignore", category=UserWarning)

# This command creates the database tables if they don't exist
models.Base.metadata.create_all(bind=engine)

# Initialize the scheduler object in the global scope
scheduler = BackgroundScheduler() 
app = FastAPI(title="Adaptive Question Platform API")

# CORS configuration
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- SCHEDULER LIFECYCLE MANAGEMENT ---

@app.on_event("startup")
def start_scheduler():
    """Starts the background scheduler when the FastAPI app starts."""
    try:
        scheduler.add_job(
            scheduled_training_job, 
            'interval', 
            minutes=1, 
        )
        if not scheduler.running:
            scheduler.start()
            print("Background scheduler started. IRT model retraining job is scheduled.")
    except Exception as e:
        print(f"Scheduler failed to start: {e}")

@app.on_event("shutdown")
def shutdown_scheduler():
    """Shuts down the background scheduler when the FastAPI app stops."""
    if scheduler.running:
        scheduler.shutdown()
        print("Background scheduler shut down.")


# --- ADMIN LOGIC: DIFFICULTY MAPPING ---
DIFFICULTY_MAPPING = {
    'easy': 0.2, 'medium': 0.5, 'hard': 0.8,       
}

def map_difficulty_to_float(label: str) -> float:
    # **Difficulty is set by ML_Predictor, this ensures the correct float value is used**
    return DIFFICULTY_MAPPING.get(label.lower(), 0.5)


# ====================================================================
# --- USER/AUTH ROUTES ---
# ====================================================================

@app.post("/login", response_model=schemas.Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), 
    db: Session = Depends(get_db)
):
    """Handles user login and issues a JWT access token."""
    user = security.authenticate_user(db, form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    # Set token expiry
    access_token_expires = timedelta(minutes=security.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    access_token = security.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me", response_model=schemas.UserOut)
def read_users_me(current_user: User = Depends(security.get_current_user)):
    """Returns the details of the current logged-in user."""
    return current_user


# FIX: Routes defined to handle both /questions/next and /questions/next/
@app.get("/questions/next", response_model=schemas.QuestionBase)
@app.get("/questions/next/", response_model=schemas.QuestionBase)
def get_next_question(
    correct_mcq_count: int = Query(0, description="Current consecutive correct MCQs. (Ignored; logic moved to ml_service)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(security.get_current_user)
):
    """Retrieves the next adaptive question for the user using ML logic."""
    
    # 1. Use the ML/Adaptive service to determine the ID of the next question
    question_id = ml_service.get_next_question_id(db, current_user.id)
    
    if question_id is None:
        raise HTTPException(status_code=404, detail="No questions found in the database.")

    # 2. Fetch the question details using the returned ID
    question = crud.get_question_by_id(db, question_id=question_id)
    
    if not question:
        raise HTTPException(status_code=404, detail=f"Question with ID {question_id} not found in database.")

    # 3. Prepare the response model
    options_str = question.options # Capture the raw string from DB
    
    # *** FIX 1: Convert DB string "null" to Python None for scenario questions ***
    if options_str and options_str.lower().strip() == 'null':
        options_str = None
        
    # Convert the comma-separated options string back to a list
    options_list = options_str.split(',') if options_str else None

    # Return the question in the schemas.QuestionBase format
    return schemas.QuestionBase(
        id=question.id,
        text=question.text,
        type=question.type,
        options=options_list, # Now correctly None for Scenarios
        difficulty=question.difficulty,
        irt_difficulty=question.irt_difficulty,
        irt_discrimination=question.irt_discrimination
    )


# ====================================================================
# --- ANSWER SUBMISSION ROUTE ---
# ====================================================================

@app.post("/questions/{question_id}/answer", response_model=schemas.AnswerResult)
def submit_answer(
    question_id: int, 
    submission: schemas.AnswerSubmit,
    db: Session = Depends(get_db),
    current_user: User = Depends(security.get_current_user)
):
    """
    Handles answer submission, checks correctness, updates performance, 
    and adjusts the user's consecutive correct MCQ count.
    """
    # 1. Fetch the question details
    question = crud.get_question_by_id(db, question_id=question_id)
    
    if not question:
        raise HTTPException(status_code=404, detail="Question not found.")

    is_correct = None
    score = None
    
    # Check if the question is an MCQ type
    if question.type.lower() == "mcq": 
        is_correct = (submission.answer.strip().lower() == question.answer.strip().lower())
        
        # Update consecutive correct MCQ count
        if is_correct:
            new_count = crud.get_correct_mcq_count(db, current_user.id) + 1
            crud.update_correct_mcq_count(db, current_user.id, new_count)
        else:
            crud.reset_correct_mcq_count(db, current_user.id)
            
    elif question.type.lower() == "scenario": 
        # *** NEW: Use NLP Service for scoring scenario questions ***
        try:
            score = nlp_service.calculate_scenario_score(
                user_answer=submission.answer, 
                correct_answer=question.answer
            )
            is_correct = score >= 0.7 
            print(f"Scenario Question {question_id} scored: {score} (Correct: {is_correct})")
        except Exception as e:
            print(f"Error calculating NLP score: {e}. Falling back to neutral score.")
            score = 0.5 
            is_correct = False
        
        crud.reset_correct_mcq_count(db, current_user.id)
        
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported question type: {question.type}")

    # 2. Update StudentPerformance (History)
    crud.update_student_performance(
        db, 
        user_id=current_user.id, 
        question_id=question_id, 
        is_correct=is_correct, 
        score=score
    )
    
    # 4. Return the result to the user
    # --- FIX 2: Return the full model answer for scenario questions ---
    correct_answer_to_display = None
    if not is_correct:
        # If wrong, show the correct answer for MCQs
        if question.type.lower() == 'mcq':
            correct_answer_to_display = question.answer
    
    # If it's a scenario, always return the model answer for detailed feedback
    if question.type.lower() == 'scenario':
        correct_answer_to_display = question.answer

    return schemas.AnswerResult(
        is_correct=is_correct,
        correct_answer=correct_answer_to_display, # <--- Uses the new logic
        score=score
    )


# ====================================================================
# --- ADMIN ROUTES (ML Difficulty setting is confirmed correct here) ---
# ====================================================================

@app.post("/admin/generate-questions", status_code=status.HTTP_200_OK)
def handle_genai_process(
    topic: str = Query(..., description="The subject topic for question generation."),
    num_questions: int = Query(10, description="Number of questions to generate (default 10)."),
    db: Session = Depends(get_db),
    current_user: User = Depends(security.get_current_user) 
) -> Dict[str, Any]:
    """Orchestrates the Gen AI -> ML Prediction -> DB Save workflow."""
    
    if not current_user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied. Must be an admin user.")

    # --- STEP 0: Deleting all existing questions ---
    print("-> STEP 0: Deleting all existing questions...")
    deleted_count = crud.delete_all_questions(db)
    print(f"   Deleted {deleted_count} existing questions.")
    
    print(f"-> STEP 1: Calling Gen AI for {num_questions} questions on topic: {topic}")
    generated_questions = generate_questions_with_ai(topic, num_questions) 
    
    if not generated_questions:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Gen AI service failed or returned empty set.")

    # List is initialized outside the loop
    refined_questions_for_db = []
    
    print(f"-> STEP 2: Running ML_Predictor on {len(generated_questions)} questions...")
    
    for question in generated_questions:
        try:
            question_text = question.get("text") or question.get("question_text") 
            
            if not question_text:
                raise ValueError("Question text field is missing.")

            # *** ML PREDICTION: Sets the actual difficulty based on the trained model ***
            predicted_label = predict_multi_class_type(question_text)
            float_difficulty = map_difficulty_to_float(predicted_label)
            # **************************************************************************
            
            question["difficulty"] = float_difficulty
            question["irt_difficulty"] = 0.0 
            question["irt_discrimination"] = 1.0 
            
            question["question_text"] = question_text 
            
            refined_questions_for_db.append(question)
            
        except Exception as e:
            print(f"ML Prediction Error on question: {e}. Using default difficulty (0.5).")
            # Fallback to default difficulty if ML fails
            question["question_text"] = question.get("text") or question.get("question_text") or "N/A"
            question["difficulty"] = 0.5 
            question["irt_difficulty"] = 0.0
            question["irt_discrimination"] = 1.0
            refined_questions_for_db.append(question)
    
    # --- STEP 3: Saving to Database (Executed ONCE) ---
    print(f"-> STEP 3: Saving {len(refined_questions_for_db)} analyzed questions to database...")
    inserted_count = crud.create_questions_bulk(db, refined_questions_for_db)
    
    if inserted_count == 0 and len(refined_questions_for_db) > 0:
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No new questions were saved to the database. Check logs for database write errors.")
        
    return {
        "status": "success",
        "message": f"Successfully deleted {deleted_count} old questions, analyzed, and saved {inserted_count} new questions on topic: '{topic}'.",
        "saved_to_db": inserted_count,
        "deleted_old_count": deleted_count
    }
@app.get("/users/progress", response_model=schemas.UserProgress)
def get_user_progress_data(
    db: Session = Depends(get_db),
    current_user: User = Depends(security.get_current_user)
):
    """
    Returns the overall progress, ability score, and recent history of the current user.
    """
    progress_data = crud.get_user_progress(db, current_user.id)
    
    if not progress_data:
        raise HTTPException(status_code=404, detail="User progress data not found.")
        
    return progress_data

@app.get("/admin/stats", response_model=schemas.AdminDashboardStats)
def get_admin_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(security.get_current_user)
):
    """
    Retrieves aggregated dashboard statistics, restricted to admin users.
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied. Must be an admin user.")
    
    stats_data = crud.get_admin_dashboard_stats(db)
    
    return stats_data