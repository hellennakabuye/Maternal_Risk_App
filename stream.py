import streamlit as st
import pandas as pd
import numpy as np
import pickle


st.write("""
# Maternal Risk Prediction App

This app predicts the **Maternal Risk** Level!
""")

st.sidebar.header('User Input Parameters')


def user_input_features():
    Age = st.sidebar.slider('Age', 10, 70, 40)
    SystolicBP = st.sidebar.slider('Systolic_BP', 70, 160, 88)
    DiastolicBP = st.sidebar.slider('Diastolic_BP', 49, 100, 69)
    BS = st.sidebar.slider('Blood Sugar', 6.0, 19.0, 7.2)
    BodyTemp = st.sidebar.slider('Body Temperature', 98.0, 103.0, 98.0)
    HeartRate = st.sidebar.slider('Heart Rate', 7, 90, 70)
    data = {'Age': Age,
            'SystolicBP': SystolicBP,
            'DiastolicBP': DiastolicBP,
            'BS': BS,
            'BodyTemp': BodyTemp,
            'HeartRate': HeartRate}
    features = pd.DataFrame(data, index=[0])
    return features


st.sidebar.markdown("""
[Example CSV input file](https://github.com/hellennakabuye/Machine-Learning/blob/main/maternal.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        Age = st.sidebar.slider('Age', 10, 70, 40)
        SystolicBP = st.sidebar.slider('Systolic_BP', 70, 160, 88)
        DiastolicBP = st.sidebar.slider('Diastolic_BP', 49, 100, 69)
        BS = st.sidebar.slider('Blood Sugar', 6.0, 19.0, 7.2)
        BodyTemp = st.sidebar.slider('Body Temperature', 98.0, 103.0, 98.0)
        HeartRate = st.sidebar.slider('Heart Rate', 7, 90, 70)
        data = {'Age': Age,
                'SystolicBP': SystolicBP,
                'DiastolicBP': DiastolicBP,
                'BS': BS,
                'BodyTemp': BodyTemp,
                'HeartRate': HeartRate}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire maternal dataset
# This will be useful for the encoding phase
maternal_raw = pd.read_csv('maternal.csv')
maternal = maternal_raw.drop(columns=['RiskLevel'])
df = pd.concat([input_df, maternal], axis=0)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('maternal_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
maternal_risks = np.array(['low risk','mid risk','high risk'])
st.write(maternal_risks[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
