import streamlit as st
import pandas as pd
import pickle

st.write("""
# Advertising Prediction App

This app predicts the **Advertising** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 16.0, 150.0, 250)
    Radio = st.sidebar.slider('Radio', 5.0,38.0 , 50.0)
    Newspaper = st.sidebar.slider('Newspaper', 40.0 ,55.0 , 65.0)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper,}

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("advertising.h5", "rb"))

prediction = loaded_model.predict(df)

st.subheader('Prediction')
st.write(prediction)
