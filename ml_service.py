import pandas as pd
import numpy as np
import joblib
import os
from sqlalchemy.orm import Session
from sqlalchemy import func
from sklearn.ensemble import RandomForestClassifier
from io import StringIO
import models
import irt_service

# Path to the trained model file
RF_MODEL_FILE = "random_forest_model.joblib"

def train_model(db: Session):
    """
    Trains and saves a RandomForestClassifier model based on student performance data.
    If no real data exists, it generates a synthetic dataset for initial training.
    """
    real_data_count = db.query(models.StudentPerformance).count()

    if real_data_count > 100:
        print("Using real data to train the model.")
        query = db.query(
            models.StudentPerformance.correct_attempts,
            models.StudentPerformance.total_attempts,
            models.User.tenth_percentage,
            models.User.twelfth_percentage,
            models.Question.difficulty,
        ).join(models.User, models.StudentPerformance.user_id == models.User.id)\
         .join(models.Question, models.StudentPerformance.question_id == models.Question.id)
        
        data = pd.read_sql(query.statement, db.bind)
        
        data['success_rate'] = data['correct_attempts'] / data['total_attempts']
        data['is_correct'] = (data['correct_attempts'] > data['total_attempts'] - data['correct_attempts']).astype(int)
        
        data = data[data['total_attempts'] > 0]
        
        if len(data) < 100:
            print("Not enough real data after filtering. Generating synthetic data.")
    else:
        print("No real data found. Generating synthetic data for initial training.")
        
        np.random.seed(42)

        num_samples = 1000
        
        difficulty = np.random.rand(num_samples) * 0.8 + 0.1
        success_rate = np.random.rand(num_samples)
        tenth_percentage = np.random.randint(50, 100, num_samples)
        twelfth_percentage = np.random.randint(50, 100, num_samples)
        
        is_correct = (success_rate > difficulty - 0.2).astype(int)

        data = pd.DataFrame({
            'difficulty': difficulty,
            'success_rate': success_rate,
            'tenth_percentage': tenth_percentage,
            'twelfth_percentage': twelfth_percentage,
            'is_correct': is_correct
        })
    
    X = data[['difficulty', 'success_rate', 'tenth_percentage', 'twelfth_percentage']]
    y = data['is_correct']
    
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        joblib.dump(model, RF_MODEL_FILE)
        print("Random Forest model trained and saved successfully.")
    except ValueError as e:
        print(f"An error occurred during training: {e}")
        print("Model training failed. This may happen with insufficient data variation.")

def get_next_question_id(db: Session, user_id: int):
    """
    Uses a probabilistic approach to select the next question, either MCQ or scenario.
    """
    # 50% chance of getting a scenario question
    if np.random.rand() > 0.7:
        scenario_question = db.query(models.Question.id).filter(models.Question.type == 'scenario').order_by(func.random()).first()
        if scenario_question:
            return scenario_question[0]
            
    # Fallback to the adaptive MCQ logic
    if not os.path.exists(RF_MODEL_FILE):
        train_model(db)
        if not os.path.exists(RF_MODEL_FILE):
            return get_random_unanswered_question(db, user_id)

    try:
        rf_model = joblib.load(RF_MODEL_FILE)
    except Exception as e:
        print(f"Failed to load RF model: {e}. Falling back to random selection.")
        return get_random_unanswered_question(db, user_id)
    
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        return None
    
    answered_ids = {q.question_id for q in db.query(models.StudentPerformance.question_id).filter_by(user_id=user_id).all()}
    unanswered_questions = db.query(models.Question).filter(models.Question.id.notin_(answered_ids), models.Question.type == 'mcq').all()
    
    if not unanswered_questions:
        return get_random_unanswered_question(db, user_id)

    question_features_list = []
    question_ids = []
    for q in unanswered_questions:
        success_rate = 0.5
        question_features_list.append([
            q.difficulty,
            success_rate,
            user.tenth_percentage, 
            user.twelfth_percentage
        ])
        question_ids.append(q.id)

    if not question_features_list:
        return get_random_unanswered_question(db, user_id)

    feature_names = ['difficulty', 'success_rate', 'tenth_percentage', 'twelfth_percentage']
    question_features_df = pd.DataFrame(question_features_list, columns=feature_names)
    
    rf_predictions = rf_model.predict_proba(question_features_df)[:, 1]
    
    combined_learning_zone_questions = []
    user_ability = user.ability_score

    for i, q in enumerate(unanswered_questions):
        rf_in_zone = 0.4 <= rf_predictions[i] <= 0.7
        irt_in_zone = abs(user_ability - q.irt_difficulty) <= 0.5 
        
        if rf_in_zone and irt_in_zone:
            combined_learning_zone_questions.append(q.id)

    if combined_learning_zone_questions:
        return int(np.random.choice(combined_learning_zone_questions))
    
    return get_random_unanswered_question(db, user_id)

def get_random_unanswered_question(db: Session, user_id: int):
    """
    A fallback function to get a random question the student hasn't answered.
    """
    answered_questions = db.query(models.StudentPerformance.question_id).filter(models.StudentPerformance.user_id == user_id).all()
    answered_ids = {q[0] for q in answered_questions}
    
    unanswered_questions = db.query(models.Question.id).filter(models.Question.id.notin_(answered_ids)).all()
    
    if unanswered_questions:
        return int(np.random.choice([q[0] for q in unanswered_questions]))
    else:
        all_questions = db.query(models.Question.id).all()
        return int(np.random.choice([q[0] for q in all_questions])) if all_questions else None