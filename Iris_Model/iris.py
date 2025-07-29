import pickle
import streamlit as st

model=pickle.load(open("flower.pkl","rb"))

def mydeploy():
    st.title("flower prediction : ")
    a = st.number_input("Enter Sepal length:", key="a")
    b = st.number_input("Enter Sepal length:", key="b")
    c = st.number_input("Enter Sepal length:", key="c")
    d = st.number_input("Enter Sepal length:", key="d")

    pred1=st.button("Predict")
    if pred1:
        pred=model.predict([[a,b,c,d]])
        
        x="flower"
        if pred==0:
            x="setosa"
        elif pred==1:
            x="versicolor"
        else:
            x="verginica"

        st.write("flower is ",x)

mydeploy()
