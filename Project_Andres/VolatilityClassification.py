import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def add_noise_to_column(df, column_name, noise_level):
    """
    Adds noise to a specific column in a DataFrame.
    
    Parameters:
        df (DataFrame): The original DataFrame.
        column_name (str): Name of the column to add noise to.
        noise_level (float): The percentage of values to flip (0 to 1).
        
    Returns:
        DataFrame: DataFrame with added noise to the specified column.
    """
    num_samples = len(df)
    num_to_flip = int(noise_level * num_samples)
    
    # Randomly choose indices to flip
    flip_indices = np.random.choice(num_samples, num_to_flip, replace=False)
    
    # Flip the values at selected indices
    noisy_column = df[column_name].copy()
    noisy_column.iloc[flip_indices] = 1 - noisy_column.iloc[flip_indices]
    
    # Update the DataFrame with the noisy column
    df[column_name + '_noisy'] = noisy_column
    
    return df


filename = 'feature_and_target.csv'
df = pd.read_csv(filename)
df_cleaned = df.dropna()



normal_data = df_cleaned[df_cleaned['risky'] == 0]
anomaly_data = df_cleaned[df_cleaned['risky'] == 1]

plt.figure(figsize=(8, 4))
plt.scatter(normal_data.index, normal_data['Volatility'], color='b', label='Normal')
plt.scatter(anomaly_data.index, anomaly_data['Volatility'], color='r', label='Risky')
plt.xlabel('Index')
plt.ylabel('Volatility')
plt.title('Volatility Data with Class labels')
plt.grid(True)
plt.legend()
plt.show()

x = df_cleaned['Volatility'].values 
y = df_cleaned['risky'].values

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size = 0.1)

df_new = pd.DataFrame({'Volatility': x_train, 'risky': y_train})
print(df_new)

noise_level = 0.2  # Example noise level (20% of values to flip)
column_name = 'risky'  # Name of the column to add noise to

noisy_df = add_noise_to_column(df_new, column_name, noise_level)
print("Original DataFrame:")
print(df_new)
print("\nDataFrame with Noisy Column:")
print(noisy_df)

x_train2 = noisy_df['Volatility'].values 
y_train2 = noisy_df['risky_noisy'].values

x_fit_train = x_train2.reshape(-1,1)
x_fit_test = x_test.reshape(-1,1)

model = LogisticRegression()
model.fit(x_fit_train, y_train)

y_pred = model.predict(x_fit_test)

#Performance results

print("Score: ",model.score(x_fit_test, y_test))

accuracy = accuracy_score(y_test, y_pred)

# Calculate precision
precision = precision_score(y_test, y_pred)

# Calculate sensitivity (recall)
sensitivity = recall_score(y_test, y_pred)

# Calculate F1 score
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Sensitivity (Recall):", sensitivity)
print("F1 Score:", f1)