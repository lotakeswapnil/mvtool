# mvtool.py
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import requests_cache
from retry_requests import retry
import openmeteo_requests
from datetime import date, timedelta
import io

st.title('Energy M&V Tool')


# --- Create session-state variable ---
if "mode" not in st.session_state:
    st.session_state.mode = None

#button1, button2, button3, button4 = st.columns(4)

# --- Display Start Buttons ---
if st.session_state.mode is None:
    st.subheader('Select Any One of the Options')

    if st.button("Enter Data (Manual)"):
        st.session_state.mode = "manual"
        st.rerun()

    if st.button("Upload Data (CSV)"):
        st.session_state.mode = "upload"
        st.rerun()


# -------------------------
# UPLOAD DATA MODE
# -------------------------

if st.session_state.mode == "upload":

    if st.button("Back to Menu"):
        st.session_state.mode = None
        st.rerun()

    st.subheader('Upload Data (CSV)')
    uploaded = st.file_uploader('', type="csv", label_visibility='collapsed')

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write('Preview:', df.head())

        # Target (dependent) column
        energy_cons = st.text_input('Dependent Variable (target column name)')

        # Number of independent vars
        num_var = st.number_input('Number of Independent Variables', min_value=1, max_value=10, step=1)

        for i in range(1,num_var+1):
            globals()[f"ind_var_{i}"] = st.text_input(f"Independent Variable {i}",key=f"var_{i}")

        model_dict = {'Linear Regression': LinearRegression, 'Ridge Regression': Ridge, 'Lasso Regression': Lasso}
        model_list = st.selectbox('Select models', model_dict)

        if st.button('Run Regression'):
            if energy_cons is not None and globals()[f"ind_var_{i}"] != "":
                if globals()[f"ind_var_{i}"] not in df.columns:
                    st.error(f"Variable '{globals()[f'ind_var_{i}']}' not found in the uploaded CSV.")
                else:
                    X = df[globals()[f'ind_var_{i}']].to_frame()
                    y = df[energy_cons]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    model = model_dict[model_list]()
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    regression = model.score(X_test, y_test)
                    cvrmse = root_mean_squared_error(y_test, preds)/y_test.mean()

                    st.write(f'Regression: {regression:.2%}')
                    st.write(f'CVRMSE: {cvrmse:.2%}')
                    st.line_chart(pd.DataFrame({'Actual': y_test, 'Predicted': preds}).reset_index(drop=True))

            else:
                st.error('All variables not defined.')


# -------------------------
# MANUAL DATA MODE
# -------------------------
elif st.session_state.mode == "manual":

    if st.button("Back to Menu"):
        st.session_state.mode = None
        st.rerun()

    st.subheader('Enter Data (Manual)')

    # Ask for number of rows & columns
    num_cols = st.number_input("Number of Dependent Variables: ", 0, 10, 3)

    # Build column names automatically
    col_names = ["Dependent Variable"]  # first column fixed
    input_valid = True  # flag to track if all names are filled

    # Generate independent variable labels
    for i in range(1, num_cols + 1):
        dependent = st.text_input(f'Independent Variable {i}:', key=f"var_{i}")

        # If blank, trigger error and mark input as invalid
        if dependent.strip() == "":
            st.error(f'Independent Variable {i} cannot be blank.')
            input_valid = False

        col_names.append(dependent)

    # Only proceed if all variable names are valid
    if input_valid:
        df_empty = pd.DataFrame("", index=range(1), columns=col_names)

        st.subheader('Enter Data Below:')
        edited_df = st.data_editor(df_empty, num_rows="dynamic")

        if st.button('Create Data'):
            st.success('Generated Data:')
            st.dataframe(edited_df)
    else:
        st.info('Please complete all Independent Variable names.')



