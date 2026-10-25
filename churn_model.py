import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# 1. Load Dataset (Using a sample simulation dataset for demonstration, 
# but you can replace this with a real Telco Churn CSV file path)
print("--- Step 1: Loading Dataset ---")
# To use a real dataset later, replace this with: df = pd.read_csv('data/telco_churn.csv')
np.random.seed(42)
n_samples = 1000
df = pd.DataFrame({
    'tenure': np.random.randint(1, 72, size=n_samples),
    'MonthlyCharges': np.random.uniform(20.0, 120.0, size=n_samples),
    'TotalCharges': np.random.uniform(20.0, 8000.0, size=n_samples),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples),
    'Churn': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
})
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.\n")

# 2. Preprocessing & Feature Engineering
print("--- Step 2: Preprocessing Data ---")
# Encode categorical variables (like Contract type) into numbers
label_enc = LabelEncoder()
df['Contract_Encoded'] = label_enc.fit_transform(df['Contract'])

# Define Features (X) and Target (y)
X = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract_Encoded']]
y = df['Churn']

# Train / Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data preprocessed and split successfully.\n")

# 3. Model Training
print("--- Step 3: Training Machine Learning Model ---")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
print("Random Forest model trained successfully.\n")

# 4. Model Evaluation
print("--- Step 4: Evaluating Model Performance ---")
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 5. Feature Importance (Shows what drives customer churn)
print("--- Step 5: Feature Importance ---")
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(feature_importances)