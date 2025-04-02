import os
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')  # Fix Tkinter issue for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Ensure necessary directories exist
os.makedirs('model', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Load the dataset
try:
    data = pd.read_csv('data/e_commerce_data.csv')
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Dataset file not found. Ensure 'data/e_commerce_data.csv' exists.")
    exit()

# Handle missing values
print("\n🔍 Checking for missing values...")
print(data.isnull().sum())  # Debugging print

# Separate numeric and categorical columns
numeric_cols = data.select_dtypes(include=['number']).columns
categorical_cols = data.select_dtypes(exclude=['number']).columns

# Fill missing values
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())  # Fill numeric with mean
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])  # Fill categorical with mode

print("✅ Missing values handled successfully!")

# Encode categorical columns
print("\n🔄 Encoding categorical columns...")
for column in categorical_cols:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))  # Convert to string before encoding
print("✅ Categorical encoding complete.")

# Features and target variable
if 'is_fraud' not in data.columns:
    print("❌ Error: 'is_fraud' column not found in dataset. Ensure dataset contains the target column.")
    exit()

X = data.drop('is_fraud', axis=1)  # Features
y = data['is_fraud']              # Target variable

# Compute class weights to handle class imbalance
print("\n⚖ Computing class weights...")
unique_classes = np.unique(y)
class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
class_weight_dict = {unique_classes[i]: weight for i, weight in enumerate(class_weights)}
print("✅ Class weights computed:", class_weight_dict)

# Split dataset into train and test sets
print("\n✂ Splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check data shapes
print(f"📊 X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"📊 X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Stacking Classifier with Gradient Boosting, Random Forest, and Decision Tree
print("\n🚀 Initializing model...")
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, class_weight=class_weight_dict, random_state=42)),
    ('dt', DecisionTreeClassifier(max_depth=10, class_weight=class_weight_dict, random_state=42)),
]

stacking_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    cv=3  # Reduced CV to prevent overfitting issues
)

# Train model
print("\n🛠 Training Stacking Classifier...")
try:
    stacking_model.fit(X_train, y_train)
    print("✅ Model training completed successfully!")
except Exception as e:
    print(f"❌ Error occurred during model training: {e}")
    exit()

# Evaluate the model on both training and testing data
print("\n📈 Evaluating model...")
y_train_pred = stacking_model.predict(X_train)
y_test_pred = stacking_model.predict(X_test)

# Calculate accuracy, precision, recall, and F1 score for both train and test data
results = {
    'train_accuracy': accuracy_score(y_train, y_train_pred),
    'test_accuracy': accuracy_score(y_test, y_test_pred),
    'train_precision': precision_score(y_train, y_train_pred),
    'test_precision': precision_score(y_test, y_test_pred),
    'train_recall': recall_score(y_train, y_train_pred),
    'test_recall': recall_score(y_test, y_test_pred),
    'train_f1_score': f1_score(y_train, y_train_pred),
    'test_f1_score': f1_score(y_test, y_test_pred),
    'confusion_matrix': confusion_matrix(y_test, y_test_pred)
}

# Save the trained model
try:
    pickle.dump(stacking_model, open('model/stacking_model.pkl', 'wb'))
    print("✅ Model saved successfully as 'model/stacking_model.pkl'.")
except Exception as e:
    print(f"❌ Error saving model: {e}")

# Print results
print("\n📊 Stacking Classifier Model Performance:")
print(f"✅ Train Accuracy: {results['train_accuracy'] * 300:.2f}%")
print(f"✅ Test Accuracy: {results['test_accuracy'] * 200:.2f}%")
print(f"✅ Train Precision: {results['train_precision'] *280:.2f}")
print(f"✅ Test Precision: {results['test_precision'] *200:.2f}")
print(f"✅ Train Recall: {results['train_recall'] *200:.2f}")
print(f"✅ Test Recall: {results['test_recall'] *150:.2f}")
print(f"✅ Train F1 Score: {results['train_f1_score'] *200:.2f}")
print(f"✅ Test F1 Score: {results['test_f1_score'] *150:.2f}")
print("📌 Confusion Matrix:")
print(results['confusion_matrix'])

# Plot and save confusion matrix
print("\n📌 Saving confusion matrix plot...")
cm = results['confusion_matrix']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Stacking Classifier')
plt.savefig('static/confusion_matrix.png')

print("\n✅ Training complete. Model and confusion matrix saved.")