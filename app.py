# -*- coding: utf-8 -*-
"""
Created on Mon Oct  17 21:01:15 2022
@author: Akash
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu


# loading the saved models

calories_model = pickle.load(open('calories_model.sav', 'rb'))



# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Calories Burnt Prediction Model',
                          
                          ['Calories Burnt Model'],
                          icons=['activity'],
                          default_index=0)
    
    
# Diabetes Prediction Page
if (selected == 'Calories Burnt Model'):
    
    # page title
    st.title('Calories Burnt Prediction Model')
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Gender = st.text_input('Gender')
        
    with col2:
        Age = st.text_input('Age')
    
    with col3:
        Height = st.text_input('Height')
    
    with col1:
        Weight = st.text_input('Weight')
    
    with col2:
        Duration = st.text_input('Duration')
    
    with col3:
        Heart_Rate = st.text_input('HeartRate')
    
    with col1:
        Body_Temp = st.text_input('Body_Temp')
    
    
    # code for Prediction
    calories_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Predicted Calories Burnt'):
        calories_prediction = calories_model.predict([['Age','Height','Duration','Heart_Rate','Body_Temp','Calories']])
        
    st.success(calories_prediction)



