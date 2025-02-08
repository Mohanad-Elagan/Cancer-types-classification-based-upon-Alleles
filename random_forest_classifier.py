import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn import svm, linear_model
from sklearn.naive_bayes import GaussianNB
from vars_local import *
import dataset

# Read dataset
df = pd.read_csv(TRAIN_PATH)
df = dataset.prepare_features(df)

X = df.drop('Cancer_Type', axis=1)
y = df['Cancer_Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Ensure all column names are strings
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

# Train a RandomForest classification model
print("RandomForest")
clf_rf = RandomForestClassifier(n_estimators=200, criterion='entropy', min_samples_split=10, verbose=True, n_jobs=-1)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
acc_score = accuracy_score(y_test, y_pred_rf)
# Output values
print("Weighted Precision:", precision_score(y_test, y_pred_rf, average='weighted'))
print("Class-wise Precision:", precision_score(y_test, y_pred_rf, average=None))
print("Accuracy Score:", acc_score)
print("Predictions:", y_pred_rf)