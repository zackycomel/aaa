import streamlit as st
import pandas as pd
import pickle

st.write("""
# Advertising Prediction App

This app predicts the **Advertising** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 0.04, 0.5, 0.6, 0.1, 0.7)
    Radio = st.sidebar.slider('Radio', 0.2, 0.7, 0.8, 0.9)
    Newspaper = st.sidebar.slider('Newspaper', 0.3,0.606, 0.5, 0.6)
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
