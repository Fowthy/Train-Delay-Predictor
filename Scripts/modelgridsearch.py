from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import pandas as pd

songs = pd.read_csv("January_Schiphol_Cleaned.csv")
print("Dataset loaded! Shape:", songs.shape)

x = songs[['feature1','feature2']]
y = songs['target']

# x and y are the features from the original dataset. x_r and y_r are the ones from the balanced dataset.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

param_grid = { 'max_depth': (2,3, 5, 7, 10, 15, 18, 22, 30, 45), 'n_estimators':(80, 100, 120, 140, 200, 300, 350, 400) }

clf = RandomForestClassifier()
kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=1)

print("Model assigned. Starting grid search...")
sh = GridSearchCV(clf, param_grid, cv=kfold)
sh.fit(x_train,y_train)

df = pd.DataFrame(sh.cv_results_)
df.to_csv('gridsearch_rfc.csv',index=False)
print("Cross Validation results saved to gridsearch_rfc.csv")

print("Best parameter combination: ",sh.best_estimator_)

