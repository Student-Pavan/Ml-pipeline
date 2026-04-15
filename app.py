from fastapi import FastAPI
import joblib
import glob
import os
PORT = int(os.environ.get("PORT", 10000))

app = FastAPI()

# Load latest model
model_files = glob.glob("model_*.pkl")
latest_model = max(model_files, key=lambda x: int(x.split("_")[1].split(".")[0]))

model = joblib.load(latest_model)

# 🔥 Add label mapping
labels = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

@app.get("/")
def home():
    return {"message": "API Running"}

@app.get("/predict")
def predict(a: float, b: float, c: float, d: float):
    result = model.predict([[a, b, c, d]])
    
    # Convert number → name
    flower_name = labels[int(result[0])]
    
    return {"prediction": flower_name}