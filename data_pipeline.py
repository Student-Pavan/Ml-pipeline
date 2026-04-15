import pandas as pd
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Save dataset
df.to_csv("dataset.csv", index=False)

print("Dataset saved as dataset.csv")