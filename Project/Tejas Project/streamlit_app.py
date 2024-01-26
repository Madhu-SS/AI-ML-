import streamlit as st
import pandas as pd
import numpy as np
import joblib


loaded_model = joblib.load('logistic_regression_model.joblib')

st.title("Choose your project name")


columns = st.columns(6)


column1_selectbox = columns[0].selectbox("industrial_risk", ["0.0", "0.5", "1.0"])
column2_selectbox = columns[1].selectbox("management_risk", ["0.0", "0.5", "1.0"])
column3_selectbox = columns[2].selectbox("financial_flexibility", ["0.0", "0.5", "1.0"])
column4_selectbox = columns[3].selectbox("credibility", ["0.0", "0.5", "1.0"])
column5_selectbox = columns[4].selectbox("competitiveness", ["0.0", "0.5", "1.0"])
column6_selectbox = columns[5].selectbox("operating_risk", ["0.0", "0.5", "1.0"])


btn = st.button("Result")


if btn:
    x =pd.DataFrame({"industrial_risk":[float(column1_selectbox)],"management_risk":[float(column2_selectbox)],"financial_flexibility":[float(column3_selectbox)],"credibility":[float(column4_selectbox)],"competitiveness":[float(column5_selectbox)],"operating_risk":[float(column6_selectbox)]})
    prediction = loaded_model.predict(x)

    if prediction == 0:
        st.write('Bankruptcy')
    else:
        st.write('Non-Bankruptcy')
