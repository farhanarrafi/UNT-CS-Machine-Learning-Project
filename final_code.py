import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,  matthews_corrcoef




filename = '3DimensionData.csv'
df = pd.read_csv(filename)
df_cleaned = df.dropna()



x = df_cleaned.iloc[:, :-1].values
y = df_cleaned['risky'].values

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size = 0.1)



model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

#Performance results

print("Score: ",model.score(x_test, y_test))

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
mcc = matthews_corrcoef(y_test, y_pred)
print(f"GaussianNB mcc: {mcc}")