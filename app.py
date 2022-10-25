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
                          
                          ['Workout Model','Calories Burnt Model'],
                          icons=['heart','activity'],
                          default_index=0)
    

if (selected == 'Workout Model'):

    # page title
    st.title('Workout Predictor')
    
    st.subheader('Hey there, this is a machine learning model for predicting a workout to hit your calories goal!')
    st.caption('Select your preferred workouts along with calorie goal and hit predict!')
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    #with col1: 
    #    Gender = 0
        #Gender = st.selectbox('Gender',('Male','Female'))
        
        #if (Gender == 'Male'):
        #    Gender = 0
         #else:
        #    Gender = 1

        
    
    with col1:
        possibleAgeList = [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
        Age = st.selectbox('Age (Years)',possibleAgeList)

            
    with col2:

        # higher corresponds to fewer and longer sets
        skinTempFactor = {'Light Walking':0.4,'Jogging':1.5,'Running':2.5,'Squats':1.3,'Push Ups':1.5,'Pull Ups':1.0
                         ,'Arm Curls':0.8,'Lateral Raises':0.8, 'Shoulder Presses':1.3, 'Deadlifts':1.0,'Bench Presses':1.0}
        
        # higher corresponds to fewer and longer sets
        setFactor = {'Light Walking':0.1,'Jogging':0.25,'Running':0.5,'Squats':2.0,'Push Ups':2.0,'Pull Ups':2.0
                         ,'Arm Curls':2.5,'Lateral Raises':2.0, 'Shoulder Presses':2.0, 'Deadlifts':2.0,'Bench Presses':2.0}
        

        Exercise = st.multiselect('Workout (5 max)',skinTempFactor.keys())

    with col3:
        possibleCalList  = [100,200,300,400,500,600]
        Calories = st.selectbox('Goal Calories (Cal)',possibleCalList)

        calPerWorkout = Calories
        if len(Exercise):
            calPerWorkout = Calories/len(Exercise)

        calorieWorkoutDict = {100:[0,2],200:[1,3],300:[2,4],400:[3,5],500:[3,5],600:[3,5]}

        for selectedCalorie in calorieWorkoutDict.keys():
            if selectedCalorie <= Calories:
                if calorieWorkoutDict[selectedCalorie][0] < len(Exercise):
                    st.error("Try increasing your calorie goal to add more workouts!")
                elif calorieWorkoutDict[selectedCalorie][1] > len(Exercise):
                    st.error("Try decreasing your calorie goal to remove workouts!")


    
    # logic
    Gender = 0
    workoutSummaryDict = {1:"",2:"",3:""}

    for workout in Exercise:

        # higher results in higher duration [0 to 1]
        durationFactor = {'Light Walking':1.0,'Jogging':0.6,'Running':0.3,'Squats':0.6,'Push Ups':0.6,'Pull Ups':0.6
                         ,'Arm Curls':0.6,'Lateral Raises':0.6, 'Shoulder Presses':0.6, 'Deadlifts':0.6,'Bench Presses':0.6}
        
        predictedDuration = round(duration_model.predict([[Gender,Age,calPerWorkout]])[0],0)
        #predictedDuration = predictedDuration*durationFactor[workout]



        workoutSet = 0
        if (predictedDuration > 5):

            # break them into sets
            predictedDuration -= skinTempFactor[workout]*(predictedDuration/5)
            workoutSet = math.floor((predictedDuration/5)*setFactor[workout])

            

        predictedHeartRate = round(HeartRange_model.predict([[Gender,Age,predictedDuration,calPerWorkout]])[0],0)

        calculatedBodyTemp = 37.5 + (predictedHeartRate/205-Age) + skinTempFactor[workout]

        predictedCalories = round(calories_model.predict([[Gender,predictedDuration,predictedHeartRate,calculatedBodyTemp]])[0],2)

        # if predcited calories are lower 
        while predictedCalories < Calories/len(Exercise):
            predictedDuration += 0.5
            predictedHeartRate = round(HeartRange_model.predict([[Gender,Age,predictedDuration,calPerWorkout]])[0],0)
            predictedCalories = round(calories_model.predict([[Gender,predictedDuration,predictedHeartRate,calculatedBodyTemp]])[0],2)

        # if predicted calories are higher 
        while predictedCalories > Calories/len(Exercise):
            predictedDuration -= 0.5
            predictedHeartRate = round(HeartRange_model.predict([[Gender,Age,predictedDuration,calPerWorkout]])[0],0)
            predictedCalories = round(calories_model.predict([[Gender,predictedDuration,predictedHeartRate,calculatedBodyTemp]])[0],2)


        if workoutSet > 0:
            workoutSummaryDict[workout] = "{} sets of {} for {} minutes each at a Heart Rate Range of {} - {} BPM. - {} Calories".format(workoutSet,workout,round(math.floor(predictedDuration/workoutSet),2),predictedHeartRate-10,predictedHeartRate+10,predictedCalories)
        else:
            workoutSummaryDict[workout] = "{} for {} minutes at a Heart Rate Range of {} - {} BPM. - {} Calories".format(workout,predictedDuration,predictedHeartRate-10,predictedHeartRate+10,predictedCalories)  


    # creating a button for Prediction

    
    if st.button('Predict Workout Duration'):

        st.success("To burn your goal of {} calories you can try: ".format(Calories))

        for summary in workoutSummaryDict:
            if workoutSummaryDict[summary]:
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
                         ,'Arm Curls':0.5,'Lateral Raises':0.7, 'Shoulder Presses':0.8, 'Deadlifts':0.5,'Bench Presses':0.8}
        
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
