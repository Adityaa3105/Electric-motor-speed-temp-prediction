# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 23:14:24 2023

@author: A
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

filename = 'trained_model.sav'
loaded_model = joblib.load('trained_model.sav')

data = pd.read_csv('D:/Data Science/DS Project/traineddata.csv')

# Function to make predictions
#def predict_speed(coolant,u_d,u_q,i_d,i_q):
  #  features = np.array([coolant,u_d,u_q,i_d,i_q])
   # prediction = loaded_model.predict(features)
   # return prediction[0]
   
def predict_speed(coolant, u_d, u_q, i_d, i_q):
    print(f"Received inputs: coolant={coolant}, u_d={u_d}, u_q={u_q}, i_d={i_d}, i_q={i_q}")
    features = np.array([coolant, u_d, u_q, i_d, i_q])
    features = features.reshape(1, -1)  # Reshape to 2D array with a single row
    print(f"Features shape: {features.shape}")
    prediction = loaded_model.predict(features)
    print(f"Prediction: {prediction}")
    return prediction[0]

# Streamlit app
def main():
    # App title
    st.title('Electric Motor Speed Prediction')
    
    coolant = st.number_input('Motor Coolant Temp',min_value=-3.0, max_value=3.0,step=0.0001)
    
    u_d = st.number_input('d component of voltage',min_value=-3.0, max_value=3.0,step=0.0001)

    u_q = st.number_input('q component of voltage',min_value=-3.0, max_value=3.0,step=0.0001)

    i_d = st.number_input('d component of current',min_value=-3.0, max_value=3.0,step=0.0001)

    i_q = st.number_input('q component of currrent',min_value=-3.0, max_value=3.0,step=0.0001)

    prediction = predict_speed(coolant,u_d,u_q,i_d,i_q)

    st.subheader('Prediction:')
    st.write(f'The predicted motor speed is: {prediction} RPM')
    

st.button('Predict', on_click=predict_speed)


if __name__ == '__main__':
    main()