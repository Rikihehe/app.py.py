#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
dftrain=pd.read_csv("Titanic_train.csv")
dftrain
dftrain=dftrain.drop('Cabin',axis=1)
dftrain_cleaned=dftrain.dropna()
dftrain_cleaned

# In[ ]:


dftrain_cleaned=dftrain_cleaned.drop(['PassengerId','Name','Ticket'], axis=1)
dftrain_cleaned

# In[ ]:


sex_type={'male':1,'female':0}
embarked_types={'S':2,'C':1,'Q':0}
dftrain_cleaned['Sex'] = dftrain_cleaned['Sex'].map(sex_type)
dftrain_cleaned['Embarked'] = dftrain_cleaned['Embarked'].map(embarked_types)
dftrain_cleaned

# In[ ]:


from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LinearRegression

x=dftrain_cleaned.drop('Survived', axis=1)
y=dftrain_cleaned['Survived']
x_train,x_test,y_train,y_test=tts(x,y,train_size=0.7,random_state=42)
x_train.shape,y_train.shape,x_test.shape,y_test.shape

# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=10000)
model.fit(x_train, y_train)

# In[ ]:


import joblib
import streamlit as st
import joblib
import numpy as np

joblib.dump(model, "logistic_regression_titanic_model.pkl")
model = joblib.load("logistic_regression_titanic_model.pkl")

sex_type = {'Male': 1, 'Female': 0}
embarked_types = {'Southampton': 2, 'Cherbourg': 1, 'Queenstown': 0}
st.title("Titanic Survival Prediction")
st.write("Enter passenger details to predict survival:")

# Inputs for the user
pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
sex = st.selectbox("Sex", list(sex_type.keys()))
age = st.slider("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare Paid", min_value=0.0, max_value=1000.0, value=50.0)
embarked = st.selectbox("Port of Embarkation", list(embarked_types.keys()))

# A prediction button
if st.button("Predict Survival"):
    input_data = np.array([
        pclass,
        sex_type[sex],
        age,
        sibsp,
        parch,
        fare,
        embarked_types[embarked]
    ]).reshape(1, -1)


    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)


    if prediction[0] == 1:
        st.success("The passenger is predicted to survive.")
    else:
        st.error("The passenger is predicted not to survive.")

    st.write(f"Prediction Confidence: {prediction_proba[0][1]:.2f}")
