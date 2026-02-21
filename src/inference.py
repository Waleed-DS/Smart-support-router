import torch
import joblib
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .text_preprocessing import clean_text
import torch.nn.functional as F

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DISTILBERT_PATH = os.path.join(MODEL_DIR, "distilbert_banking")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# Global variables
_model = None
_tokenizer = None
_label_encoder = None

def _load_resources():
    """Loads model artifacts only if they haven't been loaded yet."""
    global _model, _tokenizer, _label_encoder
    
    if _model is not None:
        return

    if not os.path.exists(DISTILBERT_PATH) or not os.listdir(DISTILBERT_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {DISTILBERT_PATH}. "
            "Please run 'notebooks/02_model_training_and_evaluation.ipynb' to generate them locally."
        )

    print("Loading models... this may take a moment.")
    try:
        _tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_PATH)
        _model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_PATH)
        _model.eval()
        _label_encoder = joblib.load(LABEL_ENCODER_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load model artifacts: {str(e)}")

def predict_intent(user_text: str) -> dict:
    """
    Production Inference Function.
    
    Logic:
    1. Preprocess text (standardize).
    2. Check HYBRID SAFETY NET (Hard Override).
       - If a high-risk keyword is found, return that intent immediately (100% Recall logic).
    3. If no keyword match, ask DistilBERT.
    """
    _load_resources()
    
    # 1. Clean the text using the Senior Pipeline
    cleaned_text = clean_text(user_text)
    
    # --- HYBRID SAFETY NET (The "Mega List" from Notebook) ---
    # We check this FIRST, just like in your hybrid_predict function.
    
    risk_map = {
        'compromised_card': [
            'hacked', 'compromised', 'unauthorized', 'suspicious', 'fraud', 
            'scam', 'phishing', 'fake', 'police', 'crime', 'victim',
            'block', 'freeze', 'lock', 'stop', 'cancel', 'protect', 
            'didn\'t make', 'did not make', 'wasn\'t me', 'was not me',
            'recognise', 'recognize', 'unknown', 'unfamiliar',
            'details', 'pin', 'cvv', 'information', 'data', 'security',
            'numbers', 'copied', 'access', 
            'someone', 'improperly', 'child', 'son', 'daughter', 'used'
        ],
        'lost_or_stolen_card': [
            'stolen', 'lost', 'robbed', 'missing', 'dropped', 'gone', 
            'thief', 'theft', 'wallet', 'purse', 'bag'
        ]
    }
    
    # Check for keywords in the cleaned text
    for intent, keywords in risk_map.items():
        if any(word in cleaned_text for word in keywords):
            return {
                "predicted_intent": intent,
                "confidence_score": 1.0, 
                "note": "Safety Override Triggered (High-Risk Keyword Found)"
            }

    # --- MODEL INFERENCE (If no keywords matched) ---
    
    # 2. Tokenize
    inputs = _tokenizer(
        cleaned_text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=64 # Matches your notebook
    )
    
    # 3. Predict
    with torch.no_grad():
        logits = _model(**inputs).logits
    
    # 4. Process Result
    probs = F.softmax(logits, dim=1)
    confidence, predicted_class_idx = torch.max(probs, dim=1)
    
    confidence_score = confidence.item()
    predicted_label = _label_encoder.inverse_transform([predicted_class_idx.item()])[0]
    
    return {
        "predicted_intent": predicted_label,
        "confidence_score": round(confidence_score, 4)
    }

if __name__ == "__main__":
    # Test cases to prove it works
    print(predict_intent("I think my son used my card without asking.")) # Should trigger Override
    print(predict_intent("What is the exchange rate for dollars?"))      # Should use Model