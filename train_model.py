from ml_service import train_model
from database import SessionLocal

if __name__ == "__main__":
    db = SessionLocal()
    try:
        # Check if a database session was successfully created
        if db:
            print("Starting ML model training...")
            train_model(db)
            print("ML model training complete.")
        else:
            print("Could not connect to the database. Please ensure PostgreSQL is running.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        # Always close the database session to free up resources
        if db:
            db.close()