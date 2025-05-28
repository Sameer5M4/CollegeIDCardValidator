import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack, csr_matrix

# Step 1: Load your dataset
df = pd.read_csv("fake_genuine_id_dataset.csv")

# Step 2: Separate columns
text_features = ['name', 'college_name', 'roll_number', 'branch']
numeric_features = ['text_match_score']
target = 'label'

# Step 3: Vectorize all text features using TF-IDF (char level)
vectorizers = {}
X_text_parts = []

for col in text_features:
    vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
    X_vec = vec.fit_transform(df[col])
    vectorizers[col] = vec
    X_text_parts.append(X_vec)

# Step 4: Scale numeric features
scaler = StandardScaler()
X_numeric = scaler.fit_transform(df[numeric_features])

# Step 5: Combine all features
X_combined = hstack(X_text_parts + [csr_matrix(X_numeric)])
y = df[target]

#split the data
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Step 6: Train the model

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
accuracy= accuracy_score(y_test,y_pred)
print("Accuray",accuracy)


print("Model trained successfully.")


# Sample test input (same fields)
test_sample = pd.DataFrame([{
    "name": "Student_10",
    "college_name": "sam",
    "roll_number": "21CS1xp234",
    "branch": "cse",
    "text_match_score": 0.9
}])

# Convert text using existing vectorizers
X_test_parts = []
for col in text_features:
    vec = vectorizers[col]
    X_vec = vec.transform(test_sample[col])
    X_test_parts.append(X_vec)

# Scale numeric field
X_test_num = scaler.transform(test_sample[numeric_features])

# Combine test data
X_test_final = hstack(X_test_parts + [csr_matrix(X_test_num)])

# Predict
y_pred = model.predict(X_test_final)
print("Predicted label:", y_pred[0])
