from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('data/e_commerce_data.csv')

# Encode categorical columns
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])

# Handle missing values
data = data.fillna(data.mean())  # Replace missing values with column mean

# Features and target variable
X = data.drop('is_fraud', axis=1)  # Features
y = data['is_fraud']              # Target variable

# Compute class weights to handle class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Stacking Classifier with Gradient Boosting, Random Forest, and Decision Tree
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, class_weight=class_weight_dict, random_state=42)),
    ('dt', DecisionTreeClassifier(max_depth=10, class_weight=class_weight_dict, random_state=42)),
]

stacking_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    cv=5
)

stacking_model.fit(X_train, y_train)

# Evaluate the model on both training and testing data
y_train_pred = stacking_model.predict(X_train)
y_test_pred = stacking_model.predict(X_test)

# Calculate accuracy, precision, recall, and F1 score for both train and test data
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

results = {
    'train_accuracy': train_accuracy,
    'test_accuracy': test_accuracy,
    'train_precision': precision_score(y_train, y_train_pred),
    'test_precision': precision_score(y_test, y_test_pred),
    'train_recall': recall_score(y_train, y_train_pred),
    'test_recall': recall_score(y_test, y_test_pred),
    'train_f1_score': f1_score(y_train, y_train_pred),
    'test_f1_score': f1_score(y_test, y_test_pred),
    'confusion_matrix': confusion_matrix(y_test, y_test_pred)
}

# Save the trained model
pickle.dump(stacking_model, open('model/stacking_model.pkl', 'wb'))

# Print results
print("Stacking Classifier Model Performance:")
print(f"Train Accuracy: {results['train_accuracy'] * 100:.2f}%")
print(f"Test Accuracy: {results['test_accuracy'] * 100:.2f}%")
print(f"Train Precision: {results['train_precision']:.2f}")
print(f"Test Precision: {results['test_precision']:.2f}")
print(f"Train Recall: {results['train_recall']:.2f}")
print(f"Test Recall: {results['test_recall']:.2f}")
print(f"Train F1 Score: {results['train_f1_score']:.2f}")
print(f"Test F1 Score: {results['test_f1_score']:.2f}")
print("Confusion Matrix:")
print(results['confusion_matrix'])

# Plot and save confusion matrix
cm = results['confusion_matrix']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Stacking Classifier')
plt.savefig('static/confusion_matrix.png')

print("Training complete. Model and confusion matrix saved.")