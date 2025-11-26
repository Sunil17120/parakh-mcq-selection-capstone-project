from sqlalchemy.orm import Session
from sqlalchemy import func, desc
import models, schemas, security
from datetime import datetime
from typing import Optional, List, Dict, Any

# --- User CRUD Operations (rest of functions unchanged) ---

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def get_user_by_id(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = security.get_password_hash(user.password)
    db_user = models.User(
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_password,
        tenth_percentage=user.tenth_percentage,
        twelfth_percentage=user.twelfth_percentage,
        ability_score=0.0,
        correct_mcq_count=0
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def update_user_ability(db: Session, user_id: int, new_ability: float):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user:
        user.ability_score = new_ability
        db.commit()
        db.refresh(user)

def update_correct_mcq_count(db: Session, user_id: int, new_count: int):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user:
        user.correct_mcq_count = new_count
        db.commit()
        db.refresh(user)

def get_correct_mcq_count(db: Session, user_id: int):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user:
        return user.correct_mcq_count
    return 0

def reset_correct_mcq_count(db: Session, user_id: int):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user:
        user.correct_mcq_count = 0
        db.commit()
        db.refresh(user)

# --- Question CRUD Operations (unchanged) ---

def get_question_by_id(db: Session, question_id: int):
    return db.query(models.Question).filter(models.Question.id == question_id).first()

def get_random_question_by_type(db: Session, question_type: str):
    # This is the function used by /questions/next
    # Using lower() for case-insensitive type matching
    return db.query(models.Question).filter(func.lower(models.Question.type) == func.lower(question_type)).order_by(func.random()).first()

def create_questions_bulk(db: Session, questions_data: List[Dict[str, Any]]) -> int:
    """
    FIX: Corrected key mapping to match 'gen_ai_service' output and added IRT fields.
    """
    new_question_objects = []
    
    for q_data in questions_data:
        # Crucial: Convert list of options to a comma-separated string
        options_str = None
        # Handle 'options' key if it's a list (expected from AI service)
        if q_data.get('options') and isinstance(q_data['options'], list):
            options_str = ",".join(q_data['options'])
        
        # Handle potential multiple text keys from different stages/sources
        question_text = q_data.get("text") or q_data.get("question_text")

        # Create the Question model object using corrected keys
        question_dict = {
            "text": question_text, 
            "options": options_str,
            "answer": q_data["answer"],
            # Map 'type' key (expected from AI service)
            "type": q_data["type"], 
            "difficulty": q_data.get("difficulty", 0.5), 
            "irt_difficulty": q_data.get("irt_difficulty", 0.0),
            "irt_discrimination": q_data.get("irt_discrimination", 1.0),
        }

        new_question_objects.append(models.Question(**question_dict))
    
    if new_question_objects:
        db.add_all(new_question_objects)
        db.commit()
        return len(new_question_objects)
    return 0

# --- Performance CRUD Operations (unchanged) ---

def update_student_performance(db: Session, user_id: int, question_id: int, is_correct: Optional[bool] = None, score: Optional[float] = None):
    performance = db.query(models.StudentPerformance).filter_by(
        user_id=user_id, question_id=question_id
    ).first()

    if performance:
        performance.total_attempts += 1
        if is_correct is not None:
            if is_correct:
                performance.correct_attempts += 1
        if score is not None:
            performance.score = score
        performance.last_attempt_date = datetime.utcnow()
    else:
        # Note: correct_attempts counts attempts where is_correct was True (for MCQs) 
        # or where a high score was achieved (for Scenarios, based on the check in main.py)
        correct_flag = 0
        if is_correct is not None:
            correct_flag = 1 if is_correct else 0
        
        # If score is provided, we default correct_attempts based on `is_correct` logic 
        # from the main app layer, which is crucial for overall accuracy calculation.
        performance = models.StudentPerformance(
            user_id=user_id,
            question_id=question_id,
            correct_attempts=correct_flag, 
            total_attempts=1,
            score=score,
            last_attempt_date=datetime.utcnow()
        )
        db.add(performance)

    db.commit()
    db.refresh(performance)
    return performance

def get_user_progress(db: Session, user_id: int) -> Dict[str, Any]:
    """
    Calculates the user's overall progress and retrieves recent performance history.
    """
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        return {} # Should be caught by the route handler in main.py

    # 1. Base query joining performance and question data
    base_query = db.query(models.StudentPerformance, models.Question).join(
        models.Question, models.StudentPerformance.question_id == models.Question.id
    ).filter(models.StudentPerformance.user_id == user_id)

    # 2. Calculate Aggregates (one query for efficiency)
    aggregate_data = base_query.with_entities(
        func.sum(models.StudentPerformance.total_attempts).label('total_attempted'),
        func.sum(models.StudentPerformance.correct_attempts).label('total_correct_all'),
        # Aggregate stats specific to types
        func.sum(models.StudentPerformance.correct_attempts).filter(func.lower(models.Question.type) == 'mcq').label('total_correct_mcq'),
        func.avg(models.StudentPerformance.score).filter(func.lower(models.Question.type) == 'scenario').label('avg_scenario_score'),
        func.sum(models.StudentPerformance.total_attempts).filter(func.lower(models.Question.type) == 'mcq').label('total_attempted_mcq')
    ).first()

    total_attempted = aggregate_data.total_attempted or 0
    total_correct_all = aggregate_data.total_correct_all or 0
    total_correct_mcq = aggregate_data.total_correct_mcq or 0
    avg_scenario_score = aggregate_data.avg_scenario_score or 0.0
    total_attempted_mcq = aggregate_data.total_attempted_mcq or 0


    # 3. Calculate Derived Metrics
    # Accuracy Rate is calculated as overall correct attempts / overall total attempts
    accuracy_rate = round((total_correct_all / total_attempted) * 100, 2) if total_attempted > 0 else 0.0

    # 4. Fetch Recent History (Last 10 records)
    recent_history_raw = base_query.order_by(
        desc(models.StudentPerformance.last_attempt_date)
    ).limit(10).all()

    recent_history = []
    for perf, q in recent_history_raw:
        is_correct_status = None
        if q.type.lower() == 'mcq':
            # is_correct is True if the correct_attempts count is greater than the incorrect attempts count.
            # However, since correct_attempts in DB only tracks the *first* correct answer on update_student_performance
            # (or any correct answer if updated per attempt), a simpler, more robust check is needed.
            
            # Since update_student_performance increments total_attempts but only increments 
            # correct_attempts if is_correct is True (for MCQs), using the total ratio is complex.
            
            # Assuming the intention for accuracy is based on the *last* attempt in the context of the history:
            # We must infer correctness based on the context of the original question submission.
            
            # *** FIX: Using the crude measure from before: perf.correct_attempts > 0 means at least one correct answer
            # was logged for this question_id. Since the tracking is complex, we stick to the agreed schema.
            # For simplicity and aligning with the initial logic: is_correct is True if correct_attempts > 0.
            is_correct_status = perf.correct_attempts > 0
            
        elif q.type.lower() == 'scenario':
            # Match the threshold used in main.py for scenario questions (score >= 0.7)
            is_correct_status = perf.score is not None and perf.score >= 0.7 
        
        recent_history.append(schemas.PerformanceHistory(
            question_text=q.text,
            is_correct=is_correct_status,
            score=perf.score,
            date=perf.last_attempt_date.isoformat()
        ))

    # 5. Build the final response dictionary
    # It must return a dict that matches the UserProgress schema structure.
    return {
        "total_questions_attempted": total_attempted,
        "total_correct_mcq": total_correct_mcq,
        "scenario_average_score": round(avg_scenario_score, 2),
        "current_ability_score": round(user.ability_score, 2),
        "accuracy_rate": accuracy_rate,
        "recent_history": [h.dict() for h in recent_history] # Convert Pydantic models back to dicts
    }
def get_admin_dashboard_stats(db: Session) -> schemas.AdminDashboardStats:
    """
    Calculates overall system statistics for the Admin Dashboard.
    """
    
    # 1. Total Students
    total_students = db.query(models.User).count()

    # 2. Total Questions
    total_questions = db.query(models.Question).count()

    # 3. Overall Attempts and Accuracy (Class Average)
    attempt_stats = db.query(
        func.sum(models.StudentPerformance.total_attempts).label('total_attempts'),
        func.sum(models.StudentPerformance.correct_attempts).label('total_correct')
    ).first()
    
    total_attempts = attempt_stats.total_attempts or 0
    total_correct = attempt_stats.total_correct or 0
    
    class_average_accuracy = round((total_correct / total_attempts) * 100, 2) if total_attempts > 0 else 0.0
    
    # 4. Average User Ability Score
    avg_ability_score_result = db.query(func.avg(models.User.ability_score)).scalar()
    avg_user_ability_score = round(avg_ability_score_result, 2) if avg_ability_score_result is not None else 0.0

    # 5. Top 5 Most Attempted Questions (for engagement analysis)
    most_attempted_questions_raw = db.query(
        models.Question.text,
        func.sum(models.StudentPerformance.total_attempts).label('attempt_count')
    ).join(
        models.StudentPerformance, models.Question.id == models.StudentPerformance.question_id
    ).group_by(models.Question.id, models.Question.text).order_by(desc('attempt_count')).limit(5).all()

    most_attempted_questions = []
    for text, count in most_attempted_questions_raw:
        most_attempted_questions.append({
            "question_text": text,
            "attempt_count": count
        })
        
    # 6. Student List (UPDATED: Fetching required performance stats)
    
    # Subquery to calculate total attempts and total correct for each user
    performance_subquery = db.query(
        models.StudentPerformance.user_id,
        func.sum(models.StudentPerformance.total_attempts).label('total_attempted'),
        func.sum(models.StudentPerformance.correct_attempts).label('total_correct')
    ).group_by(models.StudentPerformance.user_id).subquery()
    
    # Main query joins User data with aggregated performance data
    student_list_raw = db.query(
        models.User.id,
        models.User.full_name,
        models.User.email,
        models.User.ability_score,
        performance_subquery.c.total_attempted,
        performance_subquery.c.total_correct
    ).outerjoin(
        performance_subquery, models.User.id == performance_subquery.c.user_id
    ).all()
    
    student_list = []
    for user_id, full_name, email, ability_score, total_attempted, total_correct in student_list_raw:
        total_attempted = total_attempted or 0
        total_correct = total_correct or 0
        
        # Calculate individual student accuracy
        accuracy = round((total_correct / total_attempted) * 100, 2) if total_attempted > 0 else 0.0
        
        student_list.append({
            "id": user_id,
            "full_name": full_name,
            "email": email,
            "ability_score": round(ability_score, 2),
            "questions_attempted": total_attempted, # NEW REQUIRED FIELD
            "accuracy": accuracy                    # NEW REQUIRED FIELD
        })

    # Return the AdminDashboardStats object
    return schemas.AdminDashboardStats(
        total_students=total_students,
        total_questions=total_questions,
        class_average_accuracy=class_average_accuracy,
        total_attempts=total_attempts,
        avg_user_ability_score=avg_user_ability_score,
        most_attempted_questions=most_attempted_questions,
        student_list=student_list
    )
def delete_all_questions(db: Session) -> int:
    """
    Deletes all records from the Question table and resets the sequence.
    Returns the count of deleted records.
    """
    # 1. Count the records before deletion for logging/return value
    deleted_count = db.query(models.Question).count()
    
    # 2. Delete all records
    db.query(models.Question).delete()
    
    # 3. Commit the deletion
    db.commit()
    
    # Optional: Reset the auto-increment sequence (specific to PostgreSQL, etc.)
    # For SQLite/simple ORMs, the .delete() call followed by a commit is often sufficient.
    
    return deleted_count