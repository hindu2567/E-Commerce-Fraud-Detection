import pandas as pd
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the dataset
data = pd.read_csv('data/e_commerce_data.csv')

# Load the trained Stacking model
with open('model/stacking_model.pkl', 'rb') as file:
    stacking_model = pickle.load(file)

# Extract feature importance from RandomForest (a base learner in the Stacking model)
random_forest = stacking_model.named_estimators_['rf']
feature_importance = pd.DataFrame({
    'Feature': data.drop('is_fraud', axis=1).columns,
    'Importance': random_forest.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
plt.title('Feature Importance (Random Forest in Stacking)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()

# Save the plot
plt.savefig('static/feature_importance_stacking.png')
plt.close()  # Close the plot to avoid any further display issues

print("Feature importance saved as 'static/feature_importance_stacking.png'.")