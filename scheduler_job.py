# scheduler_job.py (UPDATED)
from sqlalchemy.orm import Session
# Ensure you import your database session creator
from database import SessionLocal 
# Assuming irt_service is available
import irt_service 
import ml_service 

def scheduled_training_job():
    """
    Handles the periodic retraining of IRT and ML models.
    NOTE: This function now manages its own DB session, accepting NO arguments.
    """
    print("--- Scheduled Training Job Started ---")
    
    # 1. Open a new database session within the job
    db: Session = SessionLocal()
    
    try:
        # Train IRT Model
        print("Starting IRT Model Training...")
        irt_service.train_irt_model(db)
        print("IRT Model Training Complete.")
        
        # Train Question Difficulty ML Model
        # Assuming ml_service has a function to retrain the question predictor
        print("Starting ML Predictor Training...")
        # If your ml_predictor training doesn't need DB access, call it directly:
        # ml_predictor.train_model() 
        # If it needs DB access (to load labeled data), adjust accordingly:
        # ml_service.train_question_predictor(db) 
        print("ML Predictor Training Complete.")
        
        db.commit()

    except Exception as e:
        db.rollback()
        print(f"ERROR during scheduled job: {e}")
        
    finally:
        # 2. Close the session when done
        db.close()
        print("--- Scheduled Training Job Finished ---")