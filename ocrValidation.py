# Install necessary packages:
# pip install easyocr Pillow thefuzz python-Levenshtein opencv-python-headless

import re
import easyocr
# from PIL import Image, ImageDraw, ImageFont # Not needed if not creating synthetic images
from io import BytesIO
from thefuzz import fuzz, process
import os
import csv

# --- Configuration ---
# !!! IMPORTANT: Set this to the path of your CSV file containing known college names !!!
COLLEGES_CSV_PATH = "test_college_dataset.csv"  # <--- REPLACE WITH YOUR ACTUAL CSV PATH
COLLEGE_NAME_SIMILARITY_THRESHOLD = 80
STUDENT_NAME_KEYWORDS = ["name", "student name", "student", "holder name", "name of student", "s/o", "d/o", "w/o"]
ROLL_NUMBER_KEYWORDS = ["roll no", "roll number", "id no", "reg no", "registration no", "enrollment no", "student id", "admission no", "sr no"]

NON_NAME_INDICATORS = [
    "college", "university", "institute", "vidyalaya", "polytechnic", "school", "vidyapeeth",
    "department", "dept", "branch", "faculty", "programme", "course", "stream",
    "card", "identity", "session", "academic year", "valid till", "date of birth", "dob", "issue date", "expiry date",
    "address", "city", "state", "pin", "email", "phone", "mobile", "contact",
    "principal", "director", "dean", "signature", "authorized", "controller", "examinations",
    "library", "hostel", "batch", "year", "semester", "class", "degree", "diploma", "certificate",
    "permanent", "temporary", "affiliation", "affiliated", "government", "india", "tech"
]
EXACT_LINE_EXCLUSIONS_FOR_NAME = ["identity card", "student card", "student id card", "id card"]

# --- OCR Initialization ---
OCR_READER = None
try:
    OCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
    print("EasyOCR reader initialized successfully.")
except Exception as e:
    print(f"Error initializing EasyOCR reader: {e}. OCR functionality will be limited.")

# --- Helper: Character Fixer for OCR Errors ---
def ocr_char_fixer(text, is_likely_roll_no=False):
    if not text:
        return text
    if '0' in text or '1' in text or '2' in text or '3' in text or '4' in text or \
       '5' in text or '6' in text or '7' in text or '8' in text or '9' in text or is_likely_roll_no:
        text = text.replace('O', '0').replace('o', '0')
        text = text.replace('I', '1').replace('l', '1')
        text = text.replace('S', '5').replace('s', '5')
        text = text.replace('B', '8')
        text = text.replace('Z', '2')
        if is_likely_roll_no:
            text = text.replace('G', '6')
    return text

# --- Helper: Load Known Colleges ---
def load_known_colleges(csv_path):
    colleges = []
    if not os.path.exists(csv_path):
        print(f"Warning: College names CSV not found at {csv_path}.")
        print("Please create this file or update COLLEGES_CSV_PATH in the script.")
        # Example: Create a dummy file if you want the script to run without a real one initially
        # with open(csv_path, 'w', encoding='utf-8') as f:
        #     f.write("Example University\n")
        #     f.write("Another Tech College\n")
        return [] # Return empty list if file not found and not created

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    colleges.append(row[0].strip().lower())
        print(f"Loaded {len(colleges)} colleges from {csv_path}")
    except Exception as e:
        print(f"Error loading colleges from {csv_path}: {e}")
    return colleges

# --- OCR Text Extraction ---
def extract_text_from_image_bytes(image_bytes):
    if OCR_READER is None:
        print("OCR_READER not available. Returning empty text.")
        return [], []
    try:
        results = OCR_READER.readtext(image_bytes, detail=1, paragraph=False, batch_size=4,
                                      decoder='beamsearch', beamWidth=10)
        extracted_texts = [res[1] for res in results]
        raw_ocr_results = results
        return extracted_texts, raw_ocr_results
    except Exception as e:
        print(f"Error during OCR: {e}")
        return [], []

# --- Validation Logic ---
def validate_college_name(ocr_texts, known_colleges, threshold=COLLEGE_NAME_SIMILARITY_THRESHOLD):
    if not known_colleges: return False, None, "Known colleges list is empty. Please check COLLEGES_CSV_PATH."
    if not ocr_texts: return False, None, "No text extracted for college name validation."

    best_match_score = 0
    found_college_name_in_ocr = None
    found_college_name_in_known_list = None
    searchable_texts = ocr_texts[:min(len(ocr_texts), 6)]

    for ocr_line in searchable_texts:
        ocr_line_lower = ocr_line.lower()
        if len(ocr_line_lower) < 5: continue
        match = process.extractOne(ocr_line_lower, known_colleges, scorer=fuzz.token_set_ratio)
        if match and match[1] > best_match_score:
            best_match_score = match[1]
            found_college_name_in_ocr = ocr_line
            found_college_name_in_known_list = match[0]

    for num_lines_to_combine in [2, 3]:
        if len(searchable_texts) >= num_lines_to_combine:
            combined_ocr_text = " ".join(searchable_texts[:num_lines_to_combine])
            combined_ocr_text_lower = combined_ocr_text.lower()
            if len(combined_ocr_text_lower) < 10: continue
            match = process.extractOne(combined_ocr_text_lower, known_colleges, scorer=fuzz.token_set_ratio)
            if match and match[1] > best_match_score:
                best_match_score = match[1]
                found_college_name_in_ocr = combined_ocr_text
                found_college_name_in_known_list = match[0]
    
    if best_match_score >= threshold:
        return True, found_college_name_in_known_list.title(), f"Matched '{found_college_name_in_known_list.title()}' (score {best_match_score}%) from OCR '{found_college_name_in_ocr}'."
    else:
        msg = f"College name not matched (best score: {best_match_score}% for '{found_college_name_in_known_list}' vs OCR '{found_college_name_in_ocr}')."
        return False, None, msg

def find_student_name(ocr_texts, identified_college_name_str=None):
    if not ocr_texts: return False, None, "No text for student name finding."
    name_pattern = re.compile(r"^(?:[A-Z][a-zA-Z'.\-]+)(?:\s+[A-Z][a-zA-Z'.\-]*){1,4}$")
    potential_names = []
    norm_college_name = identified_college_name_str.lower() if identified_college_name_str else ""

    for i, text_line in enumerate(ocr_texts):
        cleaned_line = text_line.strip()
        if not cleaned_line or len(cleaned_line) < 3 : continue
        line_lower = cleaned_line.lower()

        if line_lower in EXACT_LINE_EXCLUSIONS_FOR_NAME: continue
        if norm_college_name and fuzz.ratio(line_lower, norm_college_name) > 85 : continue
        
        is_header_line = False
        non_name_word_count = sum(1 for word in line_lower.split() if word in NON_NAME_INDICATORS)
        total_words = len(line_lower.split())
        if total_words > 0 and (non_name_word_count / total_words) > 0.5:
             if not any(kw.lower() in line_lower for kw in STUDENT_NAME_KEYWORDS):
                is_header_line = True
        if cleaned_line.isupper() and total_words <= 4 and non_name_word_count >=1:
             if not any(kw.lower() in line_lower for kw in STUDENT_NAME_KEYWORDS):
                is_header_line = True
        if is_header_line: continue

        found_by_keyword = False
        for kw in STUDENT_NAME_KEYWORDS:
            kw_lower = kw.lower()
            if line_lower.startswith(kw_lower):
                name_candidate_text = cleaned_line[len(kw):].strip(": ").strip()
                if name_candidate_text and name_pattern.fullmatch(name_candidate_text):
                    potential_names.append({"text": name_candidate_text, "score": 100, "method": "keyword_current_line"})
                    found_by_keyword = True; break
            if i > 0:
                prev_line_content = ocr_texts[i-1].strip().lower().rstrip(':').strip()
                if prev_line_content == kw_lower :
                    if name_pattern.fullmatch(cleaned_line):
                        potential_names.append({"text": cleaned_line, "score": 95, "method": "keyword_prev_line"})
                        found_by_keyword = True; break
        if found_by_keyword: continue

        words = cleaned_line.split()
        if not (2 <= len(words) <= 5): continue
        if name_pattern.fullmatch(cleaned_line):
            if cleaned_line.isupper() and len(cleaned_line) < 10 and not all(len(w) > 1 for w in words): continue
            if sum(c.isalpha() for c in cleaned_line.replace(" ","")) < len(cleaned_line.replace(" ","")) * 0.6: continue
            if norm_college_name and fuzz.partial_ratio(line_lower, norm_college_name) > 80:
                if len(words) <= 2 : continue
            potential_names.append({"text": cleaned_line, "score": 80, "method": "pattern_match"})

    if not potential_names:
        return False, None, "Student name not identified."
    best_candidate = sorted(potential_names, key=lambda x: x["score"], reverse=True)[0]
    return True, best_candidate["text"], f"Found name '{best_candidate['text']}' by {best_candidate['method']}."

def find_roll_number(ocr_texts):
    if not ocr_texts: return False, None, "No text for roll number finding."
    roll_number_main_patterns = [
        re.compile(r"^(?:[A-Z]{2,4}\d{2,7}[A-Z0-9]{0,7})$", re.IGNORECASE),
        re.compile(r"^(?:\d{2,4}[A-Z]{2,5}\d{2,7}[A-Z0-9]{0,2})$", re.IGNORECASE),
        re.compile(r"^(?=.*\d)[A-Z0-9]{6,15}$"),
        re.compile(r"^(?=[A-Z]*[0-9])(?:[A-Z0-9]{2,7}(?:\s?[A-Z0-9]{2,7}){1,3})$", re.IGNORECASE),
        re.compile(r"^(?:\d{7,12})$")
    ]
    just_letters_pattern = re.compile(r"^[A-Z]{5,}$")
    date_like_pattern = re.compile(r"\b(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{4})\b")
    candidate_roll_numbers = []

    for i, text_line in enumerate(ocr_texts):
        cleaned_line = text_line.strip()
        if not cleaned_line or len(cleaned_line) < 3 : continue
        if date_like_pattern.search(cleaned_line): continue
        line_lower = cleaned_line.lower()
        
        found_by_keyword = False
        for kw_idx, kw in enumerate(ROLL_NUMBER_KEYWORDS):
            kw_lower = kw.lower()
            if line_lower.startswith(kw_lower):
                potential_rn_text = cleaned_line[len(kw):].strip(": ").strip()
                if potential_rn_text:
                    normalized_rn = ocr_char_fixer(potential_rn_text.upper().replace(" ", ""), is_likely_roll_no=True)
                    if just_letters_pattern.fullmatch(normalized_rn) and len(normalized_rn) < 7: continue
                    for pattern in roll_number_main_patterns:
                        if pattern.fullmatch(normalized_rn) and len(normalized_rn) >=6 :
                            candidate_roll_numbers.append({"text": normalized_rn, "score": 100 + (10-kw_idx), "method": "keyword_current_line", "original": cleaned_line})
                            found_by_keyword = True; break
            if found_by_keyword: break
            if i > 0:
                prev_line_content = ocr_texts[i-1].strip().lower().rstrip(':').strip()
                if prev_line_content == kw_lower:
                    normalized_rn = ocr_char_fixer(cleaned_line.upper().replace(" ", ""), is_likely_roll_no=True)
                    if just_letters_pattern.fullmatch(normalized_rn) and len(normalized_rn) < 7: continue
                    for pattern in roll_number_main_patterns:
                        if pattern.fullmatch(normalized_rn) and len(normalized_rn) >=6:
                            candidate_roll_numbers.append({"text": normalized_rn, "score": 95 + (10-kw_idx), "method": "keyword_prev_line", "original": cleaned_line})
                            found_by_keyword = True; break
            if found_by_keyword: break
        if found_by_keyword: continue

        normalized_line_for_pattern = ocr_char_fixer(cleaned_line.upper().replace(" ", ""), is_likely_roll_no=True)
        if len(normalized_line_for_pattern) < 6 or len(normalized_line_for_pattern) > 15 : continue
        if just_letters_pattern.fullmatch(normalized_line_for_pattern) and len(normalized_line_for_pattern) < 7: continue

        for pattern in roll_number_main_patterns:
            if pattern.fullmatch(normalized_line_for_pattern):
                candidate_roll_numbers.append({"text": normalized_line_for_pattern, "score": 70, "method": "pattern_match_full_line", "original": cleaned_line})
                break 

    if not candidate_roll_numbers:
        return False, None, "Roll number not identified."

    unique_candidates_dict = {}
    for cand in sorted(candidate_roll_numbers, key=lambda x: x["score"], reverse=True):
        if cand["text"] not in unique_candidates_dict:
             unique_candidates_dict[cand["text"]] = cand
        elif cand["score"] > unique_candidates_dict[cand["text"]]["score"]:
             unique_candidates_dict[cand["text"]] = cand
    
    if not unique_candidates_dict: return False, None, "Roll number not identified (after dedupe)."
    best_candidate = sorted(list(unique_candidates_dict.values()), key=lambda x: x["score"], reverse=True)[0]
    return True, best_candidate["text"], f"Found roll no '{best_candidate['text']}' by {best_candidate['method']} from '{best_candidate['original']}'."

def validate_id_card_fields(image_path, known_colleges_list):
    results = {
        "college_name_found": False, "college_name_valid": False, "matched_college_name": None,
        "student_name_found": False, "extracted_student_name": None,
        "roll_number_found": False, "extracted_roll_number": None,
        "overall_status": "REJECTED", "reasons": [], "ocr_texts": []
    }
    ocr_texts = []

    if not os.path.exists(image_path):
        results["reasons"].append(f"Image file not found: {image_path}")
        return results
    if OCR_READER is None:
        results["reasons"].append("OCR Reader not initialized.")
        return results
        
    try:
        with open(image_path, 'rb') as f:
            image_bytes_content = f.read()
        ocr_texts, _ = extract_text_from_image_bytes(image_bytes_content)
    except Exception as e:
        results["reasons"].append(f"Error reading or OCR-ing image file {image_path}: {e}")
        return results # Early exit if image reading or core OCR fails

    results["ocr_texts"] = ocr_texts
    if not ocr_texts and not results["reasons"]:
        results["reasons"].append("OCR failed to extract any text from the image.")
    
    # Proceed with validation even if OCR text is empty to log failures for each step
    # The functions themselves handle empty ocr_texts

    is_college_ok, college_str, college_msg = validate_college_name(ocr_texts, known_colleges_list)
    results["college_name_found"] = bool(college_str)
    results["college_name_valid"] = is_college_ok
    results["matched_college_name"] = college_str
    if not is_college_ok: results["reasons"].append(college_msg or "College name validation failed.")

    is_name_ok, name_str, name_msg = find_student_name(ocr_texts, results["matched_college_name"])
    results["student_name_found"] = is_name_ok
    results["extracted_student_name"] = name_str
    if not is_name_ok: results["reasons"].append(name_msg or "Student name not found.")

    is_roll_ok, roll_str, roll_msg = find_roll_number(ocr_texts)
    results["roll_number_found"] = is_roll_ok
    results["extracted_roll_number"] = roll_str
    if not is_roll_ok: results["reasons"].append(roll_msg or "Roll number not found.")

    if results["college_name_valid"] and results["student_name_found"] and results["roll_number_found"]:
        results["overall_status"] = "ACCEPTED"
    else:
        results["overall_status"] = "REJECTED"
        if not results["reasons"]: # Generic reason if no specific ones were added
            results["reasons"].append("One or more required fields did not meet validation criteria.")
    return results

if __name__ == "__main__":
    print("--- ID Card Validator ---")

    # --- Configuration for the image to test ---
    # !!! IMPORTANT: REPLACE with the actual path to YOUR ID card image !!!
    image_to_validate_path = "test_samples/30.jpg"  # <--- CHANGE THIS TO YOUR IMAGE PATH

    # Load known colleges from the CSV specified in the configuration
    known_colleges = load_known_colleges(COLLEGES_CSV_PATH)

    if not os.path.exists(image_to_validate_path):
        print(f"\nERROR: Image file not found at: {image_to_validate_path}")
        print("Please update 'image_to_validate_path' in the script.")
    elif OCR_READER is None:
        print("\nERROR: Cannot process image. EasyOCR reader failed to initialize.")
    elif not known_colleges:
        print(f"\nERROR: No known colleges loaded. Please check the CSV file at '{COLLEGES_CSV_PATH}' and its content.")
    else:
        print(f"\nProcessing image: {image_to_validate_path}")
        
        # --- Optional: Print Raw OCR Output for Debugging ---
        print("\n--- Raw OCR Output ---")
        try:
            with open(image_to_validate_path, 'rb') as f_img:
                img_bytes = f_img.read()
            raw_ocr, _ = extract_text_from_image_bytes(img_bytes)
            if raw_ocr:
                for i, line_txt in enumerate(raw_ocr):
                    print(f"  OCR Line {i}: '{line_txt}'")
            else:
                print("  No text extracted by OCR.")
        except Exception as e_ocr:
            print(f"  Error during raw OCR extraction for debug: {e_ocr}")
        print("--- End of Raw OCR Output ---\n")
        # --- End of Optional Debugging Section ---

        validation_results = validate_id_card_fields(image_to_validate_path, known_colleges)
        
        print("\n--- Validation Results ---")
        print(f"  Overall Status: {validation_results['overall_status']}")
        print(f"  Matched College Name: {validation_results['matched_college_name']} (Valid: {validation_results['college_name_valid']})")
        print(f"  Extracted Student Name: {validation_results['extracted_student_name']} (Found: {validation_results['student_name_found']})")
        print(f"  Extracted Roll Number: {validation_results['extracted_roll_number']} (Found: {validation_results['roll_number_found']})")
        
        if validation_results["reasons"]:
            print("\n  Reasons/Messages for status:")
            for reason_msg in validation_results["reasons"]:
                print(f"    - {reason_msg}")
        
        # print("\n  Full details (including all OCR'd text):")
        # for res_key, res_value in validation_results.items():
        #     if res_key == "ocr_texts" and isinstance(res_value, list):
        #         print(f"  {res_key}:")
        #         for ocr_idx, ocr_line_text in enumerate(res_value):
        #             print(f"    Line {ocr_idx}: {ocr_line_text}")
        #     else:
        #         print(f"  {res_key}: {res_value}")