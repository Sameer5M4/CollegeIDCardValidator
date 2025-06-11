import cv2
import numpy as np
import easyocr
import re
import os
import csv
from thefuzz import fuzz, process

# --- Configuration ---
COLLEGES_CSV_PATH = "collegeDataSet.csv"
COLLEGE_NAME_SIMILARITY_THRESHOLD = 85  # Slightly more lenient for real-world OCR

# --- Load Resources (once, when the server starts) ---
def load_known_colleges(csv_path):
    colleges = []
    if not os.path.exists(csv_path):
        print(f"FATAL ERROR: College names CSV not found at {csv_path}.")
        return []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row: colleges.append(row[0].strip().lower())
        print(f"Loaded {len(colleges)} colleges from {csv_path}")
        return colleges
    except Exception as e:
        print(f"Error loading colleges from {csv_path}: {e}")
        return []

print("Initializing EasyOCR reader...")
OCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
KNOWN_COLLEGES = load_known_colleges(COLLEGES_CSV_PATH)
print("OCR resources loaded.")


# --- Main Function to be called by FastAPI ---
def run_fields_check(image_np: np.ndarray) -> dict:
    """
    Validates text fields on an ID card from a NumPy image array.
    Returns a dictionary with a validation score and details.
    """
    if not OCR_READER or not KNOWN_COLLEGES:
        return {"score": 0.0, "details": "OCR resources not loaded."}
    
    try:
        # Convert NumPy array to bytes for EasyOCR
        is_success, buffer = cv2.imencode(".jpg", image_np)
        if not is_success:
            return {"score": 0.0, "details": "Failed to encode image for OCR."}
        
        image_bytes = buffer.tobytes()
        
        # This is a simplified version of your logic, focused on scoring
        ocr_results = OCR_READER.readtext(image_bytes, detail=0, paragraph=True)
        
        if not ocr_results:
            return {"score": 0.0, "details": "OCR failed to extract any text."}
        
        # --- Scoring Logic ---
        # 1. College Name Score (most important)
        college_score = 0.0
        college_details = "College name not found."
        full_text = " ".join(ocr_results).lower()
        
        match = process.extractOne(full_text, KNOWN_COLLEGES, scorer=fuzz.token_set_ratio)
        if match and match[1] >= COLLEGE_NAME_SIMILARITY_THRESHOLD:
            college_score = match[1] / 100.0  # Normalize score to 0.0-1.0
            college_details = f"College matched '{match[0].title()}' with score {match[1]}%."
        elif match:
            college_details = f"Best college match '{match[0].title()}' was below threshold (Score: {match[1]}%)."

        # 2. Name and Roll Number Score (simple presence check)
        name_found = any(re.search(r'\b(name|student)\b', text, re.I) for text in ocr_results)
        roll_found = any(re.search(r'\b(roll|reg|id|enrollment)\b', text, re.I) for text in ocr_results)
        
        name_score = 1.0 if name_found else 0.0
        roll_score = 1.0 if roll_found else 0.0

        # 3. Final Weighted Score
        # Weights: College=60%, Name=20%, Roll Number=20%
        final_score = (college_score * 0.6) + (name_score * 0.2) + (roll_score * 0.2)
        
        details_list = [college_details]
        if not name_found: details_list.append("Student name keyword not found.")
        if not roll_found: details_list.append("Roll number keyword not found.")
        
        return {
            "score": final_score,
            "details": " | ".join(details_list)
        }

    except Exception as e:
        print(f"ERROR in field validation: {e}")
        return {"score": 0.0, "details": str(e)}