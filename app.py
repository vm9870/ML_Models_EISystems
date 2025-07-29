
import streamlit as st
import pickle
import numpy as np
from sklearn.datasets import load_iris

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def areaprice_model_ui():
    st.header("Area vs Price Prediction")
    area = st.number_input("Enter area (in sq. ft.):", min_value=100)
    if st.button("Predict Price"):
        model = load_model("Areaprice_Model/model1.pkl")
        price = model.predict([[area]])[0]
        st.success(f"Predicted Price: ₹{int(price)}")

def insurance_model_ui():
    st.header("Insurance Cost Predictor")
    age = st.slider("Enter Age:", 18, 100, 25)
    if st.button("Predict Insurance Purchase"):
        model = load_model("Insurance_Model/model2.pkl")
        pred = model.predict([[age]])[0]
        result = "Will Buy Insurance" if pred == 1 else "Will Not Buy Insurance"
        st.success(result)

def multi_model_ui():
    st.header("House Price Predictor (Multi Features)")
    area = st.number_input("Area (sq ft):", min_value=100)
    bedrooms = st.slider("Bedrooms:", 1, 10, 3)
    age = st.slider("House Age (years):", 0, 100, 10)
    if st.button("Predict House Price"):
        model = load_model("Multi_Model/model3.pkl")
        price = model.predict([[area, bedrooms, age]])[0]
        st.success(f"Predicted Price: ₹{int(price)}")

def iris_model_ui():
    st.header("Iris Flower Classifier")
    sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1)
    sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.5)
    petal_length = st.slider("Petal Length", 1.0, 7.0, 1.4)
    petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2)
    if st.button("Predict Species"):
        model = load_model("Iris_Model/model4.pkl")
        data = [[sepal_length, sepal_width, petal_length, petal_width]]
        pred = model.predict(data)[0]
        iris = load_iris()
        st.success(f"Predicted Species: {iris.target_names[pred]}")

st.title("ML Models Dashboard")

option = st.selectbox("Choose a model to test:", [
    "Area vs Price", 
    "Insurance Prediction", 
    "Multi Feature Price Prediction", 
    "Iris Flower Classification"
])

if option == "Area vs Price":
    areaprice_model_ui()
elif option == "Insurance Prediction":
    insurance_model_ui()
elif option == "Multi Feature Price Prediction":
    multi_model_ui()
else:
    iris_model_ui()
