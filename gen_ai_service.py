# gen_ai_service.py (Full corrected version with manual schema and robust JSON parsing)

import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from os import environ
from google import genai
from google.genai import types
import re # <-- Added back for JSON cleanup

# --- Structured Output Schema (Pydantic - UNCHANGED) ---
class GeneratedQuestion(BaseModel):
    text: str = Field(description="The question text.")
    options: Optional[List[str]] = Field(default=None, description="A list of 4 options for MCQ, or None for Scenario questions.")
    answer: str = Field(description="The correct answer for MCQ, or a detailed model answer for Scenario questions.")
    type: str = Field(description="The type of question, must be 'mcq' or 'scenario'.")

class QuestionBatch(BaseModel):
    questions: List[GeneratedQuestion] = Field(max_items=10, min_items=1)
# -------------------------------------------

# Function updated to use robust schema definition
def generate_questions_with_ai(topic: str, num_questions: int = 10) -> Optional[List[Dict[str, Any]]]:
    """
    Calls the Gen AI model to generate questions with structured output.
    Uses direct types.Schema definition to avoid Pydantic V2 conversion errors and includes
    error-tolerant JSON parsing.
    """
    api_key = environ.get('GEMINI_API_KEY')
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set. Cannot call Gen AI service.")

    # --- (Prompt setup) ---
    prompt = f"""
    Generate exactly {num_questions} questions on the topic: '{topic}'.
    Mix the question types between 'mcq' (Multiple Choice Question with 4 options) and 'scenario' (A short hypothetical situation/case study).
    The output MUST strictly follow the QuestionBatch JSON schema.
    For 'mcq' questions, 'options' must be a list of exactly 4 strings.
    For 'scenario' questions, 'options' must be set to null or None.
    Ensure the 'answer' for scenario questions is a detailed model answer/solution.
    """
    
    # 1. Manually define the Gemini Schema for a single GeneratedQuestion
    question_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "text": types.Schema(type=types.Type.STRING),
            "options": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
                description="List of 4 options for MCQ, or null for scenario."
            ),
            "answer": types.Schema(type=types.Type.STRING),
            "type": types.Schema(type=types.Type.STRING, description="Must be 'mcq' or 'scenario'."),
        },
        required=["text", "answer", "type"]
    )
    
    # 2. Define the Gemini Schema for the overall QuestionBatch (output_schema)
    output_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "questions": types.Schema(
                type=types.Type.ARRAY,
                items=question_schema,
                description=f"A list of {num_questions} generated questions."
            )
        },
        required=["questions"]
    )
    
    try:
        client = genai.Client(api_key=api_key) 
        
        # 3. Define the structured output config
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=output_schema, 
        )
        
        print(f"Calling Gemini to generate {num_questions} questions on '{topic}'...")
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=config,
        )

        if response.text:
            raw_text = response.text
            
            try:
                # 4. Standard attempt to parse JSON
                data = json.loads(raw_text)
                return data.get('questions')
                
            except json.JSONDecodeError as json_e:
                # 5. If parsing fails, attempt to sanitize and re-parse
                print(f"JSON Decode Error detected: {json_e}. Attempting cleanup...")
                
                # Strip leading/trailing non-JSON characters (e.g., Markdown fence)
                # Note: The re.sub pattern works across multiple lines due to re.DOTALL
                cleaned_text = re.sub(r'^\s*```json\s*|^\s*```\s*|\s*```\s*$', '', raw_text, flags=re.IGNORECASE | re.DOTALL).strip()
                
                # Replace unescaped newlines within strings (common cause of 'Unterminated string')
                # This replacement assumes the model output a newline inside a property value.
                cleaned_text = cleaned_text.replace('\n', '\\n') 
                
                try:
                    # Final attempt to parse the cleaned text
                    data = json.loads(cleaned_text)
                    print("JSON successfully repaired and parsed.")
                    return data.get('questions')
                except json.JSONDecodeError as repair_e:
                    # If repair fails, log both errors and re-raise the issue as a service error
                    print(f"Failed to repair JSON: {repair_e}")
                    raise Exception(f"Failed to parse and repair JSON response. Original error: {json_e}")

        else:
            print("Gemini returned an empty response text.")
            return None
        
    except Exception as e:
        print(f"Gen AI Service Error: {e}")
        return None