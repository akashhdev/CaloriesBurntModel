# -*- coding: utf-8 -*-
"""
Created on Mon Oct  17 21:01:15 2022
@author: Akash Raj Patel
"""

import pickle
import streamlit as st
import math
from streamlit_option_menu import option_menu


# loading the saved models

calories_model = pickle.load(open('calories_model.sav', 'rb'))
HeartRange_model = pickle.load(open('HeartRange_model.sav', 'rb'))
duration_model = pickle.load(open('duration_model.sav','rb'))

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Workout Planner Systems',
                          
                          ['Workout Duration Model','Calories Burnt Model'],
                          icons=['heart','activity'],
                          default_index=0)
    

if (selected == 'Workout Duration Model'):

    # page title
    st.title('Workout Duration Predictor')
    
    st.subheader('Hey there, this is a machine learning model for predicting the duration of a workout to hit your calories goal!')
    st.caption('Enter the details of your workout and calorie goal to get the rough duration and recommended heart rate range.')
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1: 
        Gender = st.selectbox('Gender',('Male','Female'))
        
        if (Gender == 'Male'):
            Gender = 0
        else:
            Gender = 1
    
    with col2:
        possibleAgeList = [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
        Age = st.selectbox('Age (Years)',possibleAgeList)

            
    with col3:
        workout_factor = {'Light Walking':0.4,'Jogging':0.7,'Running':1.5,'Cycling':2.0,'Squats':1.2,'Push Ups':1.3,'Pull Ups':1.5
                         ,'Arm Curls':0.8,'Lateral Raises':0.8, 'Shoulder Presses':1.3, 'Deadlifts':1.0,'BenchPresses':1.0}
        
        Exercise = st.multiselect('Workout (5 max)',workout_factor.keys())
        
        if len(Exercise) > 5:
            st.error("You can only select 3 workouts max right now")

    with col1: 
        Calories = st.number_input('Goal Calories (Cal)')

        if Calories < 0:
            st.error("WE'RE HERE TO LOSE CALORIES MAN!")

        calPerWorkout = Calories
        if len(Exercise):
            calPerWorkout = Calories/len(Exercise)

    
    # logic

    workoutSummaryDict = {1:"",2:"",3:""}

    for workout in Exercise:

        predictedDuration = round(duration_model.predict([[Gender,Age,calPerWorkout]])[0],0)

        workoutSet = 0
        if (predictedDuration > 5):
            workoutSet = math.floor(predictedDuration/5)
            predictedDuration -= workout_factor[workout]*(predictedDuration/5)

        predictedHeartRate = round(HeartRange_model.predict([[Gender,Age,predictedDuration,calPerWorkout]])[0],0)

        
        
        if workoutSet > 0:
            workoutSummaryDict[workout] = "{} sets of {} for {} minutes each at a Heart Rate Range of {} - {} BPM.".format(workoutSet,workout,math.floor(predictedDuration/workoutSet),predictedHeartRate-10,predictedHeartRate+10)
        else:
            workoutSummaryDict[workout] = "{} for {} minutes at a Heart Rate Range of {} - {} BPM.".format(workout,predictedDuration,predictedHeartRate-10,predictedHeartRate+10)  


    # creating a button for Prediction

    
    if st.button('Predict Workout Duration'):

        st.success("To burn your goal of {} calories you can try: ".format(Calories))

        for summary in workoutSummaryDict:
            st.success(workoutSummaryDict[summary])

        if (summary):
            st.success("Take a rest of {} seconds between each set.".format(45))
        
    else: 
        st.success('Enter the details and press the predict button!')

      
# Calories Prediction Page
elif (selected == 'Calories Burnt Model'):
    
    # page title
    st.title('Calories Burnt Predictor')
    
    st.subheader('Hey there, this is a machine learning model used for predicting calories burnt during a workout!')
    st.caption('Enter the details of your workout and get the rough amount of calories you will burn!')
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Gender = st.selectbox('Gender',('Male','Female'))
        
        if (Gender == 'Male'):
            Gender = 0
        else:
            Gender = 1
            
    with col2:
        Heart_Range = {"90-110":100,"110-130":120,"130-150":140,"150-180":170}
        Heart_Rate = st.selectbox('Target Heart Range',Heart_Range.keys())
        Heart_Rate = Heart_Range[Heart_Rate]
        
    with col3: 
        workout_factor = {'Light Walking':0.4,'Jogging':0.7,'Running':1.4,'Cycling':1.4,'Squats':1.2,'Push Ups':1.2,'Pull Ups':1.2
                         ,'Arm Curls':0.5,'Lateral Raises':0.7, 'Shoulder Presses':0.8, 'Deadlifts':0.5,'BenchPresses':0.8}
        
        Exercise = st.selectbox('Workout',workout_factor.keys())
        
    with col1:
        Duration = st.number_input('Duration (Minutes)')

    Body_Temp = 37.5 + (Heart_Rate/180) + workout_factor[Exercise]

    
    # code for Prediction
    calories_predicted = 'Enter the details and press the predict button!'
    
    # creating a button for Prediction
    
    if st.button('Predicted Calories Burnt'):
        calories_predicted = "You will burn around {} Calories by doing {} minutes of {}.".format(round(calories_model.predict([[Gender,Duration,Heart_Rate,Body_Temp]])[0],2), Duration, Exercise)
        
    st.success(calories_predicted)


st.caption('Made with ❤ By Akash Raj Patel')
st.caption('Website: https://quib.dev')
st.caption('Github: https://github.com/QuibDev/CaloriesBurntModel')
st.caption('Database: https://www.kaggle.com/datasets/fmendes/fmendesdat263xdemos')
