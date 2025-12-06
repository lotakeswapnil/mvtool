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

energy_cons = st.text_input('Target column name (energy usage)')
num_var = st.number_input('Number of variables', min_value=1, max_value=10, step=1)
for i in range(1,num_var+1):
    globals()[f"ind_var_{i}"] = st.text_input(
        f"Independent variable {i}",
        key=f"var_{i}"
    )

if energy_cons is not None and globals()[f"ind_var_{i}"] != "" :
    X = df[globals()[f'ind_var_{i}']].to_frame()
    y = df[energy_cons]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    regression = model.score(X_test, y_test)

    st.write(f'Regression: {regression:.2%}')
    st.line_chart(pd.DataFrame({'Actual': y_test, 'Predicted': preds}).reset_index(drop=True))

else if energy_cons is not None and globals()[f"ind_var_{i}"] in df.columns :
    st.write('Variable not found.')

else:
    st.write('All Variables not defined.')