import re
import string

def clean_text(text: str) -> str:
    """
    Standardizes text using the 'Senior Pipeline' logic found in training.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Domain Specific Normalization
    abbrev_map = {
        r"\batm\b": "atm",
        r"\bpin\b": "pin",
        r"\bcard lost\b": "lost card",
        r"\bstolen\b": "stolen"
    }
    for pattern, replacement in abbrev_map.items():
        text = re.sub(pattern, replacement, text)
    
    # 3. Mask Numbers
    text = re.sub(r'\d+', '<NUM>', text)

    # 4. Remove punctuation (preserve currency/hash)
    preserve = "$%#"
    all_punct = string.punctuation
    trans_table = str.maketrans('', '', ''.join(c for c in all_punct if c not in preserve))
    text = text.translate(trans_table)

    # 5. Whitespace
    text = " ".join(text.split())
    
    return text