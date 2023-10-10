#you first need to download the data from teams and put it in the same folder as this to be able to load the script
#the time is in duration (timedelta) format so that needs to be changed in the future
#here i load the libraries
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
sns.set_theme()

df= pd.read_csv('January_Schiphol_Time_Groups.csv')
df

StationDummies=pd.get_dummies(df['Station_Name'])
StationDummies

DayDummies=pd.get_dummies(df['day_of_week'])
DayDummies

TimeDummies=pd.get_dummies(df['Time_Group'])
TimeDummies

TrainDummies=pd.get_dummies(df['Train_Type'])
TrainDummies

FinalDF=pd.concat([df, StationDummies, DayDummies, TimeDummies, TrainDummies], axis=1)
FinalDF

FinalDF['DelayOrNot'] = np.where(FinalDF['Train_Delay']<=60, 0, 1).astype(int)
FinalDF.info()

#then i import the function to split train and test data, select the y and x features and split the data into test and train data
from sklearn.model_selection import train_test_split
#then i set the x and y axis and devide the data in train and test data.
X = FinalDF.iloc[:, [12, 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70]]
y = FinalDF['DelayOrNot']

from collections import Counter
from imblearn.under_sampling import NearMiss

nm = NearMiss()
x_nm, y_nm = nm.fit_resample(X, y)
print(Counter(y))
print(Counter(y_nm))

X_train, X_test, y_train, y_test = train_test_split(x_nm, y_nm, test_size=0.2, random_state = 42)
#here i look at the information from the new dataset
FinalDF.info()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

parameters1 = {'criterion':('gini', 'entropy'), 'max_depth': [2,3,4,5,6,7,8,9,10,11,12,13], 'n_estimators':[30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]}

clf_GS = GridSearchCV(RandomForestClassifier(), parameters1)
clf_GS.fit(x_nm, y_nm)

BestCriterion = clf_GS.best_estimator_.get_params()['criterion']
BestMax_depth = clf_GS.best_estimator_.get_params()['max_depth']
BestN_estimators = clf_GS.best_estimator_.get_params()['n_estimators']

dfBestRF_Over=pd.DataFrame([[BestCriterion,BestMax_depth,BestN_estimators]],columns=['Best criterion','Best max_depth','Best n_estimators'])
dfBestRF_Over.to_csv('GridSearchRandomForest5.csv')