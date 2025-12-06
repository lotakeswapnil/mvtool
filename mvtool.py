# app.py
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.title('Energy M&V — Simple Regression Demo')

uploaded = st.file_uploader('Upload CSV (features + target)', type='csv')

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write('Preview:', df.head())

#target_col = st.text_input('Target column name (energy usage)')

if target_col:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    regression = model.score(X_test, y_test)

    st.write(f'Regression: {regression:.2%}')
    st.line_chart(pd.DataFrame({'Actual': y_test, 'Predicted': preds}).reset_index(drop=True))
