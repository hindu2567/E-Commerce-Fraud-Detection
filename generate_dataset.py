import pandas as pd
import numpy as np

# Generate sample data
np.random.seed(42)

# Number of rows to generate
num_rows =10

# Generate data
data = {
    'Transaction ID': [f'TR{1000+i}' for i in range(num_rows)],
    'Customer ID': np.random.randint(1000, 2000, size=num_rows),
    'Transaction Amount': np.random.uniform(10, 5000, size=num_rows),  # Random amount between 10 and 5000
    'Payment Method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Bank Transfer'], size=num_rows),
    'Product Category': np.random.choice(['Electronics', 'Clothing', 'Home & Kitchen', 'Books'], size=num_rows),
    'Quantity': np.random.randint(1, 10, size=num_rows),
    'Customer Age': np.random.randint(18, 70, size=num_rows),
    'Device Used': np.random.choice([0, 1], size=num_rows),  # 0 could mean mobile, 1 could mean desktop
    'Account Age (days)': np.random.randint(30, 365, size=num_rows),
    'Transaction Hour': np.random.randint(0, 24, size=num_rows),  # Random hour between 0 and 23
    
}

# Create DataFrame
df = pd.DataFrame(data)

# Create a synthetic 'is_fraud' column based on some conditions
df['is_fraud'] = np.random.choice([0, 1], size=num_rows)  # Random 0 or 1 (0 for no match, 1 for match)


# Save to CSV
df.to_csv('data/e_commerce_data.csv', index=False)

# Check the first few rows of the synthetic dataset
print(df.head())