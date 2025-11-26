import spacy
# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Please download the 'en_core_web_md' model by running:")
    print("python -m spacy download en_core_web_md")
    exit()

def get_keywords(text):
    """
    Extracts key nouns and verbs from a text.
    """
    doc = nlp(text.lower())
    return [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ']]

def calculate_scenario_score(user_answer: str, correct_answer: str) -> float:
    """
    Calculates a comprehensive similarity score using semantic and keyword analysis.
    """
    if not user_answer or user_answer.strip() == "":
        return 0.0

    if not correct_answer or correct_answer.strip() == "":
        return 0.0

    user_doc = nlp(user_answer.lower())
    correct_doc = nlp(correct_answer.lower())

    # Step 1: Semantic Similarity Score
    semantic_score = user_doc.similarity(correct_doc)
    
    # Step 2: Keyword and Concept Matching
    correct_keywords = set(get_keywords(correct_answer))
    user_keywords = set(get_keywords(user_answer))

    matched_keywords = correct_keywords.intersection(user_keywords)
    keyword_match_score = len(matched_keywords) / len(correct_keywords) if correct_keywords else 0.0

    # Step 3: Syntactic and Structural Analysis (a simple check)
    # This is a basic example; a more complex check would be needed for production.
    structural_score = 1.0 if len(user_answer.split()) > 5 else 0.5 

    # Step 4: Combine scores with weights
    # Adjust weights based on importance
    # Example weights: 60% semantic, 30% keyword, 10% structural
    final_score = (0.6 * semantic_score) + (0.3 * keyword_match_score) + (0.1 * structural_score)
    
    # Ensure the score is within the valid range [0, 1]
    return round(max(0.0, min(1.0, final_score)), 2)