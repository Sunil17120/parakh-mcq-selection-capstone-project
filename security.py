# security.py

from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from typing import Optional # <-- NEW: Required for Optional[timedelta]
import crud, database, models
import os

# Password Hashing Setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Configuration
# Reading from environment variable (Best Practice)
SECRET_KEY = os.getenv("SECRET_KEY", "your-very-secure-fallback-key-that-should-be-replaced") 
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 # Default expiry

# This scheme will extract the token from the "Authorization: Bearer <token>" header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

# --- FIX: Added expires_delta to function signature and implemented its logic ---
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    
    # Calculate expiration time based on delta or default minutes
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        # Fallback to default minutes if no delta is provided
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
# -------------------------------------------------------------------------------

# Dependency to get the current logged-in user from a token
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(database.get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = crud.get_user_by_email(db, email=email)
    if user is None:
        raise credentials_exception
    return user
    
def authenticate_user(db: Session, email: str, password: str):
    # 1. Fetch user by email
    user = crud.get_user_by_email(db, email=email)
    if not user:
        return False
    # 2. Verify the password hash
    if not verify_password(password, user.hashed_password):
        return False
        
    return user