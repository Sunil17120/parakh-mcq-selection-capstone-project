import pandas as pd
import numpy as np
import string
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

# --- Configuration ---
# NOTE: Update this to point to your actual combined Q&A data file
DATA_FILE_PATH = 'C:/Users/Sunil Kumar Pandab/Desktop/studentapp/backend/merged_output.txt' 

# Define the paths for saving the trained components (Job Files/Artifacts)
MODEL_DIR = 'model_artifacts'
MODEL_PATH = os.path.join(MODEL_DIR, 'multi_class_model.keras')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

# Global variables for loaded components (will be loaded on demand for prediction)
vectorizer_multi = None
label_encoder_multi = None
model_multi = None

# --- Text Cleaning Function ---
def clean_text(text):
    """Converts text to lowercase and removes punctuation."""
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# --- Training and Saving Function ---

def train_and_save_model(file_path=DATA_FILE_PATH):
    """Loads data, trains the multi-class model and saves all components."""
    global vectorizer_multi, label_encoder_multi, model_multi

    print("--- Starting Question Difficulty Model Training Job ---")

    # 1. Load Data
    try:
        df = pd.read_csv(file_path, sep='\t')
    except FileNotFoundError:
        print(f"Error: Required training data file '{file_path}' not found. Aborting training.")
        return

    # 2. Prepare Multi-class Dataset
    questions_multi_df = df[['Question', 'DifficultyFromQuestioner']].copy()
    questions_multi_df.rename(columns={'Question': 'text', 'DifficultyFromQuestioner': 'label'}, inplace=True)
    answers_multi_df = df[['Answer']].copy()
    answers_multi_df.rename(columns={'Answer': 'text'}, inplace=True)
    answers_multi_df['label'] = 'not a question'
    
    combined_multi_df = pd.concat([questions_multi_df, answers_multi_df], ignore_index=True)
    combined_multi_df['cleaned_text'] = combined_multi_df['text'].apply(clean_text)
    
    label_encoder_multi = LabelEncoder()
    combined_multi_df['label'] = combined_multi_df['label'].fillna('missing_difficulty')
    combined_multi_df['encoded_label'] = label_encoder_multi.fit_transform(combined_multi_df['label'])
    
    # 3. Feature Engineering (TF-IDF Vectorization)
    vectorizer_multi = TfidfVectorizer(max_features=5000) 
    tfidf_features_multi = vectorizer_multi.fit_transform(combined_multi_df['cleaned_text'])
    
    # 4. Build and Train Multi-class Model
    num_classes = combined_multi_df['encoded_label'].nunique()
    num_features_multi = tfidf_features_multi.shape[1]
    
    model_multi = Sequential([
        Input(shape=(num_features_multi,)), 
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax') 
    ])
    
    model_multi.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    X_train_multi, X_val_multi, y_train_multi, y_val_multi = train_test_split(
        tfidf_features_multi, combined_multi_df['encoded_label'], test_size=0.2, random_state=42
    )
    
    model_multi.fit(X_train_multi, y_train_multi, epochs=5, batch_size=32, verbose=1, validation_data=(X_val_multi, y_val_multi))
    
    print("\nTraining complete.")

    # 5. Save the trained components (Job File/Artifacts)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    try:
        model_multi.save(MODEL_PATH)
        with open(VECTORIZER_PATH, 'wb') as f:
            pickle.dump(vectorizer_multi, f)
        with open(ENCODER_PATH, 'wb') as f:
            pickle.dump(label_encoder_multi, f)
        print(f"ML Model artifacts saved successfully to '{MODEL_DIR}'.")
    except Exception as e:
        print(f"Error saving model artifacts: {e}")

# --- Loading Function ---

def load_model_components():
    """Loads the trained components from disk."""
    global vectorizer_multi, label_encoder_multi, model_multi

    if not os.path.exists(MODEL_PATH):
        print("Model artifacts not found. Please run this file directly to train and save the model first.")
        return False

    try:
        model_multi = load_model(MODEL_PATH)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer_multi = pickle.load(f)
        with open(ENCODER_PATH, 'rb') as f:
            label_encoder_multi = pickle.load(f)
            
        print("ML Model components loaded successfully for prediction.")
        return True
        
    except Exception as e:
        print(f"Error loading model components: {e}")
        return False

# --- Final Prediction Function ---

def predict_multi_class_type(text: str) -> str:
    """
    Predicts the class ('easy', 'medium', 'hard', 'not a question') for the input text.
    Loads components automatically if they haven't been loaded.
    """
    global vectorizer_multi, model_multi, label_encoder_multi
    
    # Load components if they are not in memory (first call)
    if model_multi is None and not load_model_components():
        return "Model not initialized/loaded"
        
    cleaned_text = clean_text(text)
    
    tfidf_features_single = vectorizer_multi.transform([cleaned_text])
    
    prediction_probabilities = model_multi.predict(tfidf_features_single, verbose=0)[0]
    predicted_class_index = np.argmax(prediction_probabilities)
    
    predicted_label = label_encoder_multi.inverse_transform([predicted_class_index])[0]
    
    return predicted_label

if __name__ == '__main__':
    # When this file is run directly, it executes the training and saving process.
    train_and_save_model()