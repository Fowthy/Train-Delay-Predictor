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

from sklearn.model_selection import train_test_split
from sklearn.utils import resample

Delay=FinalDF[FinalDF['DelayOrNot']==1]
NoDelay=FinalDF[FinalDF['DelayOrNot']==0]
DelayCount=Delay.Station_Code.count()
NoDelayCount=NoDelay.Station_Code.count()

NoDelay_Under=NoDelay.sample(DelayCount, random_state=42)
FinalDF_under=pd.concat([NoDelay_Under, Delay]).reset_index()
train_X2 = FinalDF_under.iloc[:, [12, 17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71]]
train_y2 = FinalDF_under['DelayOrNot']
X_train_train2, X_test_train2, y_train_train2, y_test_train2 = train_test_split(train_X2, train_y2, test_size=0.2, random_state = 42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

parameters1 = {'criterion':('gini', 'entropy'), 'max_depth': [2,3,4,5,6,7,8,9,10,11,12,13], 'n_estimators':[30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]}

clf_GS1 = GridSearchCV(RandomForestClassifier(), parameters1)
clf_GS1.fit(train_X2, train_y2)

BestCriterion = clf_GS1.best_estimator_.get_params()['criterion']
BestMax_depth = clf_GS1.best_estimator_.get_params()['max_depth']
BestN_estimators = clf_GS1.best_estimator_.get_params()['n_estimators']

dfBestRF_Over=pd.DataFrame([[BestCriterion,BestMax_depth,BestN_estimators]],columns=['Best criterion','Best max_depth','Best n_estimators'])
dfBestRF_Over.to_csv('GridSearchRandomForest4.csv')