from sqlalchemy.orm import Session
from sqlalchemy import select, text
import csv
import models
from database import SessionLocal, engine
# Assumes ml_service contains the train_model function for student performance
from ml_service import train_model

# --- Import ML components from the new module ---
try:
    from ml_predictor import predict_multi_class_type
    print("Successfully imported Question Difficulty prediction module.")
except ImportError as e:
    print(f"Error importing ml_predictor: {e}. Ensure ml_predictor.py is available.")
    # Fallback to prevent runtime errors if import fails
    def predict_multi_class_type(text): return "medium" 
# ------------------------------------------------

# Make sure your models are loaded and tables are created
models.Base.metadata.create_all(bind=engine)

## --- Model-to-Database Difficulty Mapping ---

# Map the string labels from the ML model to the required float range (0.1 - 0.9)
DIFFICULTY_MAPPING = {
    'easy': 0.2,       # Low difficulty
    'medium': 0.5,     # Mid difficulty
    'hard': 0.8,       # High difficulty
}

def map_difficulty_to_float(label: str) -> float:
    """Maps a string difficulty label to a float value (0.1 - 0.9)."""
    return DIFFICULTY_MAPPING.get(label.lower(), 0.5)

def refine_questions_with_model(question_data_list: list) -> list:
    """
    Applies the ML model to classify text, filters out 'not a question', 
    and maps valid questions to float difficulty.
    """
    print("\nStarting Question Difficulty classification and filtering...")
    refined_data = []
    skipped_count = 0
    
    for q_data in question_data_list:
        text = q_data['text']
        
        # 1. Get the predicted difficulty label (loads model components automatically)
        predicted_label = predict_multi_class_type(text)
        
        # 2. FILTER STEP: Skip if classified as 'not a question' or a training artifact label
        if predicted_label.lower() in ['not a question', 'missing_difficulty', 'model not initialized/loaded']:
            skipped_count += 1
            continue
            
        # 3. MAP STEP: Convert the predicted label to the required float difficulty
        float_difficulty = map_difficulty_to_float(predicted_label)
        
        # Update the question data with the new, refined difficulty
        q_data['difficulty'] = float_difficulty
        
        refined_data.append(q_data)
        
    print(f"Classification and filtering complete. Skipped {skipped_count} non-questions or invalid entries.")
    return refined_data


def seed_database():
    db: Session = SessionLocal()
    
    questions_to_add = []
    csv_file_path = 'questions.csv'

    try:
        # Clear existing data
        db.execute(text("TRUNCATE TABLE questions, student_performance RESTART IDENTITY CASCADE;"))
        db.commit()
        print("Successfully cleared existing questions and student performance data.")
        
        # 1. Read questions from CSV
        with open(csv_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Store data, the 'difficulty' field will be overwritten by ML model
                questions_to_add.append({
                    "text": row["text"],
                    "options": row["options"],
                    "answer": row["answer"],
                    "difficulty": 0.5, 
                    "type": row["type"]
                })

        if not questions_to_add:
            print("No questions found in CSV file. Seeding aborted.")
            return

        # 2. Refine questions using the ML model (filters and updates difficulty)
        refined_questions = refine_questions_with_model(questions_to_add)

        # 3. Add valid questions to Database
        new_questions = []
        for q_data in refined_questions:
            exists = db.scalar(select(models.Question).filter_by(text=q_data["text"]))
            
            if not exists:
                new_question = models.Question(**q_data)
                new_questions.append(new_question)
        
        if new_questions:
            db.add_all(new_questions)
            db.commit()
            print(f"\nSuccessfully added {len(new_questions)} new questions to the database.")
            
            # Train the student performance model (using your existing function in ml_service.py)
            print("Training student performance model via ml_service...")
            train_model(db)
            print("Student performance model training complete.")
        else:
            print("No new questions to add after ML filtering. Database remains unchanged.")

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found. Please create it with your questions.")
        db.rollback()
    except Exception as e:
        db.rollback()
        print(f"An unexpected error occurred: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()