# -*- coding: utf-8 -*-
"""
Created on Sun May  8 21:01:15 2022
@author: siddhardhan
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
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction using ML')
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Gender = st.text_input('Number of Pregnancies')
        
    with col2:
        Age = st.text_input('Glucose Level')
    
    with col3:
        Height = st.text_input('Blood Pressure value')
    
    with col1:
        Weight = st.text_input('Skin Thickness value')
    
    with col2:
        Duration = st.text_input('Insulin Level')
    
    with col3:
        Heart_Rate = st.text_input('BMI value')
    
    with col1:
        Body_Temp = st.text_input('Diabetes Pedigree Function value')
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        calories_prediction = calories_model.predict([['Age','Height','Duration','Heart_Rate','Body_Temp','Calories']])
        
        if (calories_prediction[0] == 1):
          calories_diagnosis = 'The person is diabetic'
        else:
          calories_diagnosis = 'The person is not diabetic'
        
    st.success(calories_diagnosis)



