# imports

import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns                       # data lib
from sklearn.model_selection import train_test_split 
from xgboost import XGBRegressor           # model used
from sklearn import metrics 

# loading data

calories = pd.read_csv('data/calories.csv')
exercise = pd.read_csv('data/exercise.csv')


# checking data

#print('\nCalories\n')
#print(calories.head())
#print(calories.shape)


#print('\nExercise\n')
#print(exercise.head())
#print(exercise.shape)

# combining both dataframes

calories_data = pd.concat([exercise,calories['Calories']],axis=1) 
# axis = 1 means adding data row wise 

#checking for missing values
calories_data.isnull().sum()


## Data Analysis 

# getting some statistical measures about data 
calories_data.describe()

# converting text data to numerical values
calories_data.replace({'Gender':{'male':0,'female':1}}, inplace=True)



## Data Visualization 

# plotting the gender column in count plot 

sns.set()

# plotting the gender column in count plot 
sns.countplot(calories_data['Gender'])

# plotting distribution plot of age 
sns.displot(calories_data['Age'])

# plotting distribution of weight   
sns.displot(calories_data['Weight'])


## Finding Corelation between data

correlation = calories_data.corr()


# plotting corelation heatmap 

plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')

# --> calories and duration have the highest corelation (1.0)



# Separating features and target (calories)
X = calories_data.drop(columns=['User_ID','Calories'],axis=1)
Y = calories_data['Calories']



## Splitting test and train data 
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state = 2)


## Training model

# loading mode
model = XGBRegressor()

# training model 
model.fit(X_train,y_train)
 
 
## Evaluation 
test_data_prediction = model.predict(X_test)


mae = metrics.mean_absolute_error(y_test, test_data_prediction)


print("\n\nmae: ",mae)







