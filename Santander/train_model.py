import pandas as pd
import os
os.chdir('/home/tomas/Kaggle/Santander')

from clean_data import clean_data
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

#Import samples
df = pd.read_csv('data/sample.csv', low_memory=False)

df_clean = clean_data(df)
#Split columns into train data and labels
data = df_clean.ix[:,:12]       
labels = df_clean.ix[:,12:]


#Split into trainig, validation and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)


#Normailze data
classifier = Pipeline([
    #Normalizer
    ('clf', OneVsRestClassifier(KNeighborsClassifier()))
    ])



bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X, y)

#KNN Classifier
clf = OneVsRestClassifier(KNeighborsClassifier())