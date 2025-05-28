import pandas as pd
import pytesseract
import re
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack, csr_matrix

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ============ STEP 1: TRAIN MODEL ============

df = pd.read_csv("fake_genuine_id_dataset.csv")  # Replace with new file once uploaded

text_features = ['name', 'college_name', 'roll_number', 'branch']
numeric_features = ['text_match_score']
target = 'label'

vectorizers = {}
X_text_parts = []

for col in text_features:
    vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
    vectorizers[col] = vec
    X_text_parts.append(vec.fit_transform(df[col]))

scaler = StandardScaler()
X_numeric = scaler.fit_transform(df[numeric_features])
X_combined = hstack(X_text_parts + [csr_matrix(X_numeric)])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("✅ Model trained. Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# ============ STEP 2: OCR AND FIELD PARSING ============

def extract_text(img_path):
    img = Image.open(img_path)
    return pytesseract.image_to_string(img)

def detect_field_type(line):
    line = line.strip().lower()

    # Match college name (keywords)
    if re.search(r'(college|university|institute|engineering)', line):
        return 'college_name'

    # Match roll number (e.g., 22CS1056, 239X5A05M4)
    if re.search(r'roll\s*no[:\s]*[a-z0-9]+', line) or re.match(r'^[a-z0-9]{6,}$', line.replace(" ", ""), re.IGNORECASE):
        return 'roll_number'

    # Match branch (like "Branch: B.Tech (CSE)" or just "CSE")
    if re.search(r'\bbranch\b', line) and re.search(r'(cse|ece|mech|civil|it|eee)', line):
        return 'branch'
    if re.match(r'^(cse|ece|mech|civil|it|eee)$', line.strip(), re.IGNORECASE):
        return 'branch'

    # Match name (2+ alphabetic words, likely name)
    if re.match(r'^[a-zA-Z]{2,}(?:\s+[a-zA-Z]{2,})+$', line):
        return 'name'

    return None


def build_field_dict(ocr_lines):
    assigned = {}
    used_lines = set()

    for line in ocr_lines:
        if not line.strip(): continue
        predicted_field = detect_field_type(line)
        if predicted_field and predicted_field not in assigned:
            assigned[predicted_field] = line.strip()
            used_lines.add(line.strip())

    return assigned

# ============ STEP 3: PREDICT WITH FINAL MODEL ============

def predict_with_model(field_data):
    for key in text_features:
        field_data.setdefault(key, "")

    field_data['text_match_score'] = 0.85  # default confidence

    X_parts = []
    for col in text_features:
        X_parts.append(vectorizers[col].transform([field_data[col]]))
    X_numeric = scaler.transform([[field_data['text_match_score']]])
    X_input = hstack(X_parts + [csr_matrix(X_numeric)])
    return model.predict(X_input)[0]

# ============ STEP 4: EXECUTE PIPELINE ============

img_path = "test_samples/1.jpg"  # Replace with actual path to test image
ocr_result = extract_text(img_path)
ocr_lines = ocr_result.strip().split("\n")

print("\n🔍 OCR Lines:")
for line in ocr_lines:
    if line.strip():
        print("•", line.strip())

field_dict = build_field_dict(ocr_lines)

print("\n📦 Fields Detected:")
for k, v in field_dict.items():
    print(f"{k}: {v}")

prediction = predict_with_model(field_dict)
print(f"\n✅ Final ID Prediction: {str(prediction).upper()}")
