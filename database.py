import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Use an environment variable for the database password for better security and flexibility
# You will need to set this environment variable in your terminal before running the app.
# Example: export DATABASE_PASSWORD="your_password"
database_password = os.getenv("DATABASE_PASSWORD", "sunil")
DATABASE_URL = f"postgresql://postgres:{database_password}@localhost/student_auth_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency to get a DB session for each request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
