import joblib
import time
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

 
import pandas as pd

df = pd.read_csv("dataset.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Predictions
pred = model.predict(X)
acc = accuracy_score(y, pred)

# Save model with version
filename = f"model_{int(time.time())}.pkl"
joblib.dump(model, filename)

print("Model saved as:", filename)
print("Accuracy:", acc)

# Fail pipeline if accuracy is low
if acc < 0.9:
    raise Exception("Model accuracy too low!")

# Save logs
with open("training_log.txt", "a") as f:
    f.write(f"Model: {filename}, Accuracy: {acc}\n")