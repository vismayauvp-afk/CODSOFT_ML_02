# fraud_detection_user_input.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ===== Step 1: Load dataset =====
print("Loading dataset...")
data = pd.read_csv("fraudTest.csv")  # make sure this CSV is in the same folder

# ===== Step 2: Select only numeric features we want for input =====
features_to_use = ['amt', 'lat', 'long', 'merch_lat', 'merch_long', 'city_pop']
X = data[features_to_use]
y = data['is_fraud']

print("Splitting data...")
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Applying SMOTE to handle class imbalance...")
# Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Training model...")
# Train Random Forest
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train_res, y_train_res)

print("Model trained successfully!")

# ===== Step 3: Evaluate model =====
y_pred = model.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===== Step 4: User Input Prediction =====
print("\nEnter transaction details to predict fraud:")

amt = float(input("Transaction amount: "))
lat = float(input("Customer latitude: "))
long = float(input("Customer longitude: "))
merch_lat = float(input("Merchant latitude: "))
merch_long = float(input("Merchant longitude: "))
city_pop = int(input("City population: "))

user_input = pd.DataFrame([{
    'amt': amt,
    'lat': lat,
    'long': long,
    'merch_lat': merch_lat,
    'merch_long': merch_long,
    'city_pop': city_pop
}])

prediction = model.predict(user_input)[0]

print("\n🔍 Prediction Result:")
if prediction == 1:
    print("🚨 FRAUD TRANSACTION DETECTED")
else:
    print("✅ LEGITIMATE TRANSACTION")
