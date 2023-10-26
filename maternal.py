import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
import pickle


maternal = pd.read_csv('maternal.csv')

df = maternal.copy()
target = 'RiskLevel'

target_mapper = {'low risk':0, 'mid risk':1, 'high risk':2}


def target_encode(val):
    return target_mapper[val]


df['RiskLevel'] = df['RiskLevel'].apply(target_encode)

X = df.drop('RiskLevel', axis=1)
Y = df['RiskLevel']

# Build random forest model
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
pickle.dump(clf, open('maternal_clf.pkl', 'wb'))
