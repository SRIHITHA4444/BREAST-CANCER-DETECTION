import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib  
from sklearn.preprocessing import StandardScaler

# Neural Network class 
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# Load data to get feature names
data = pd.read_csv("breast_cancer_detection.csv")
X = data.drop(columns = ["id", "Unnamed: 32", "diagnosis"])  # assuming 'diagnosis' is the label
feature_names = X.columns.tolist()

# Load the trained model
input_size = len(feature_names)
model = NeuralNetwork(input_size = input_size, hidden_size = 64, output_size = 1)
model.load_state_dict(torch.load("breast_cancer_model.pth", map_location = torch.device('cpu')))
model.eval()

# Load the scaler if used during training
scaler = joblib.load("scaler.save")

# Streamlit UI
st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")
st.title("🧬 Breast Cancer Predictor")
st.markdown("Provide the following inputs to predict if the tumor is **Malignant** or **Benign**:")

# Input fields
user_input = []
for feature in feature_names:
    val = st.number_input(f"{feature}", value=float(X[feature].mean()), format = "%.4f", step = 0.0001)
    user_input.append(val)

# Predict button
if st.button("Predict"):
    input_df = pd.DataFrame([user_input], columns=feature_names)

    # Scale the input
    input_scaled = scaler.transform(input_df)  
    # Convert to tensor
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32) 
    # Get prediction
    with torch.no_grad():
        prediction = model(input_tensor).item()
        result = "Malignant" if prediction >= 0.5 else "Benign"
        st.subheader(f"🔍 Prediction: **{result}**")
        st.write(f"Confidence Score: `{prediction:.4f}`")
