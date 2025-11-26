from database import SessionLocal, engine
import models
import security
import sys

# Ensure tables exist
models.Base.metadata.create_all(bind=engine)

def create_super_admin():
    db = SessionLocal()
    
    email = input("Enter Admin Email: ")
    password = input("Enter Admin Password: ")
    full_name = "System Admin"

    # Check if user exists
    existing_user = db.query(models.User).filter(models.User.email == email).first()
    
    if existing_user:
        print(f"User {email} already exists. Updating to Admin...")
        existing_user.is_admin = True
        # Optional: Update password if needed
        # existing_user.hashed_password = security.get_password_hash(password) 
    else:
        print("Creating new Admin user...")
        hashed_password = security.get_password_hash(password)
        new_admin = models.User(
            email=email,
            full_name=full_name,
            hashed_password=hashed_password,
            tenth_percentage=0,   # Not relevant for admin
            twelfth_percentage=0, # Not relevant for admin
            ability_score=0.0,
            is_admin=True         # <--- IMPORTANT
        )
        db.add(new_admin)
    
    db.commit()
    print("Admin created successfully!")
    db.close()

if __name__ == "__main__":
    create_super_admin()