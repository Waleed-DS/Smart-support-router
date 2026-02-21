import sys
import os

# Ensure we can find the src folder
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from src.inference import predict_intent
    print(" Successfully imported 'predict_intent' from src/inference.py")
except ImportError as e:
    print(f" Import Failed: {e}")
    exit(1)

# TEST 1: Normal Query (Should use DistilBERT)
print("\n--- Test 1: Normal Banking Query ---")
try:
    normal_result = predict_intent("What is the exchange rate for dollars?")
    print(f"Input: 'What is the exchange rate for dollars?'")
    print(f"Output: {normal_result}")
    
    if "note" not in normal_result:
        print(" Test 1 Passed (Model Inference used)")
    else:
        print(" Test 1 Warning: Safety logic triggered unexpectedly.")
except Exception as e:
    print(f" Test 1 Failed: {e}")

# TEST 2: Safety Override (Should use Hybrid Logic)
print("\n--- Test 2: High-Risk Safety Trigger ---")
try:
    # Using a keyword from your 'Mega List' ("son" + "used")
    risk_result = predict_intent("I think my son used my card without asking.")
    print(f"Input: 'I think my son used my card without asking.'")
    print(f"Output: {risk_result}")
    
    if risk_result.get("confidence_score") == 1.0 and "Safety Override" in risk_result.get("note", ""):
        print(" Test 2 Passed (Hybrid Safety Net triggered correctly)")
    else:
        print(" Test 2 Failed: Safety override did NOT trigger.")
except Exception as e:
    print(f" Test 2 Failed: {e}")