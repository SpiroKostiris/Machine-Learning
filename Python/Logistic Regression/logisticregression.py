import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

X = pd.read_csv('train.csv');
y = X.pop("Survived");

# Format categories into numbers
gender = {'male': 1,'female': 0}
X['Sex'] = [gender[item] for item in X['Sex']]
X["Embarked"].fillna('O',inplace=True)
embark = {'C': 1, 'S': 2, 'Q': 3,'O': 0}
X['Embarked'] = [embark[item] for item in X['Embarked']]

# Remove columns Survived, PassengerId, Name, Ticket and Cabin.
# Reasons: First attribute is y value, next 3 attributes don't provide useful information to give an accurate prediction(i.e.little to no dependency on survival). Last variable has too many missing entries in its column
X.pop("PassengerId")
X.pop("Name")
X.pop("Ticket")
X.pop("Cabin")

# Use average to replace missing values
X["Age"].fillna(X.Age.mean(),inplace=True)

# Told to use 80:20 ratio rule
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
logreg = LogisticRegression(C=100, intercept_scaling=1, dual=False, fit_intercept=True, penalty='l2',max_iter=200, tol=0.0001)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of Logistic Regression model classifier on test set from train.csv: {:.2f}'.format(logreg.score(X_test, y_test)))

forestmodel = RandomForestClassifier(n_estimators=200)
forestmodel.fit(X_train, y_train)
y_pred = forestmodel.predict(X_test)
print('Accuracy of Random Forrest classifier on test set from train.csv: {:.2f}'.format(forestmodel.score(X_test, y_test)))

# Predict test.csv results
X_pred = pd.read_csv('test.csv');
X_pred['Sex'] = [gender[item] for item in X_pred['Sex']]
X_pred["Embarked"].fillna('O',inplace=True)
X_pred['Embarked'] = [embark[item] for item in X_pred['Embarked']]
idnum = X_pred.pop("PassengerId")
X_pred.pop("Name")
X_pred.pop("Ticket")
X_pred.pop("Cabin")

# Use average to replace missing values
X_pred["Age"].fillna(X_pred.Age.mean(),inplace=True)
X_pred = X_pred.fillna(X_pred.mean()).copy()
y_pred = logreg.predict(X_pred)

submission = pd.DataFrame({
                          "PassengerId": idnum,
                          "Survived": y_pred
})
submission.to_csv('TestPredictionsLogistic.csv', index=False);

y_pred = forestmodel.predict(X_pred)

submission = pd.DataFrame({
                          "PassengerId": idnum,
                          "Survived": y_pred
                          })
submission.to_csv('TestPredictionsForest.csv', index=False);







