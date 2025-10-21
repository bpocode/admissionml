import streamlit as st
import pickle
import numpy as np
import pandas as pd

password_guess = st.text_input("What is the Password?")
if password_guess != st.secrets["password"]:
    st.stop


# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

with open('reg_admission.pickle', 'rb') as f:
    clf = pickle.load(f)
    
st.title('Admissions Predictor') 
st.image('admission.jpg', width = 800)

st.sidebar.header('**Admission Chances input**')
gre = st.sidebar.slider('GRE Score', min_value=0,max_value=300,step=1)
toefl = st.sidebar.slider('TOEFL Score', min_value=0, max_value=100, step=1)
gpa = st.sidebar.slider('GPA', min_value=0.0, max_value=4.0, step=0.1)
research = st.sidebar.selectbox('Research Experience', options = ['Yes', 'No'])
rating = st.sidebar.slider('University Rating', min_value=0, max_value=5, step=1)
sop = st.sidebar.slider('Statement of Purpose(SOP)', min_value=0.0, max_value=5.0, step=0.5)
lor = st.sidebar.slider('Letter of Recommendation(LOR)', min_value=0.0, max_value=5.0, step=0.5)

Research_No, Research_Yes = 0,0
if research == 'Yes':
    Research_yes = 1
if research == 'No':
    Research_No = 1



if st.sidebar.button("Predict"):
    # Build input row in the same order the model was trained on.
    X = [[gre, toefl, rating, sop, lor, gpa, Research_No, Research_Yes]]
    y_test_pred, y_test_pis = clf.predict(X, alpha=0.1)

print(y_test_pred)
print(y_test_pis)



