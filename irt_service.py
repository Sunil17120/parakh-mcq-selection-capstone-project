import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func
import models
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_irt_model(db: Session):
    """
    Trains the IRT model. 
    1. Updates Question params (Difficulty/Discrimination).
    2. Updates User params (Ability Score).
    """
    logger.info("Starting IRT training process...")

    # --- PART 1: UPDATE QUESTIONS (Synthetic Warm-up) ---
    questions = db.query(models.Question).all()
    if not questions:
        return

    real_question_ids = [q.id for q in questions]
    question_id_map = {index: q_id for index, q_id in enumerate(real_question_ids)}
    num_questions = len(real_question_ids)

    # Use synthetic data to ensure questions always have difficulty values
    # even if no one has answered them yet.
    try:
        import girth
        # Synthetic setup
        synthetic_difficulty = np.random.randn(num_questions)
        synthetic_discrimination = np.random.rand(num_questions) + 0.5
        synthetic_thetas = np.random.randn(200)
        synthetic_responses = girth.create_synthetic_irt_dichotomous(
            synthetic_difficulty, synthetic_discrimination, synthetic_thetas
        )
        results = girth.twopl_mml(synthetic_responses)
        
        new_diffs = results['Difficulty']
        new_discs = results['Discrimination']

        # Update Questions in DB
        for index, diff in enumerate(new_diffs):
            db_id = question_id_map.get(index)
            if db_id is not None:
                q = db.query(models.Question).filter(models.Question.id == db_id).first()
                if q:
                    q.irt_difficulty = float(diff)
                    q.irt_discrimination = float(new_discs[index])
        
        db.commit()
        logger.info("Questions calibrated using synthetic warm-up.")

    except ImportError:
        logger.warning("GIRTH not installed. Skipping advanced calibration.")
    except Exception as e:
        logger.error(f"Error calibrating questions: {e}")

    # --- PART 2: UPDATE USERS (The Missing Link) ---
    # We calculate a simple ability score based on performance
    logger.info("Calculating User Ability Scores...")
    
    users = db.query(models.User).filter(models.User.is_admin == False).all()
    
    for user in users:
        # Get all performance records for this user
        perfs = db.query(models.StudentPerformance).filter_by(user_id=user.id).all()
        
        if not perfs:
            continue

        # Simple IRT Estimation Logic:
        # Ability increases if you answer hard questions correctly.
        # Ability decreases if you answer easy questions incorrectly.
        
        total_score = 0.0
        weight_sum = 0.0
        
        for p in perfs:
            # Get the question's difficulty
            question = db.query(models.Question).filter(models.Question.id == p.question_id).first()
            if not question:
                continue
            
            diff = question.irt_difficulty
            
            # If Correct: You get points proportional to difficulty
            # (Hard question correct = More points)
            if p.correct_attempts > 0:
                total_score += (diff + 2.0) # Shift range to be positive
            else:
                # If Wrong: You get points, but fewer (Easy question wrong = Penalty)
                total_score += (diff - 1.0) # Penalty
                
            weight_sum += 1

        # Calculate Average
        if weight_sum > 0:
            raw_ability = total_score / weight_sum
            
            # Clamp ability between -3 and +3 (Standard IRT range)
            final_ability = max(-3.0, min(3.0, raw_ability))
            
            user.ability_score = final_ability
            logger.info(f"Updated User {user.email} ability to {final_ability}")

    db.commit()
    logger.info("User ability scores updated.")