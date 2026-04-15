import streamlit as st
import joblib
import glob

st.title("ML Model Dashboard")

# Load latest model
model_files = glob.glob("model_*.pkl")
latest_model = max(model_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
model = joblib.load(latest_model)

st.write("Using Model:", latest_model)

# Input fields
a = st.number_input("Feature A", value=5.1)
b = st.number_input("Feature B", value=3.5)
c = st.number_input("Feature C", value=1.4)
d = st.number_input("Feature D", value=0.2)

# Label mapping
labels = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

if st.button("Predict"):
    prediction = model.predict([[a, b, c, d]])
    
    flower_name = labels[int(prediction[0])]
    
    st.success(f"Prediction: {flower_name}")