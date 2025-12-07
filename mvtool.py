# mvtool.py
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

st.title('Energy M&V — Simple Regression Demo')


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

    # Run button
    if st.button("Run Regression"):

        # 1) Validate that target is provided
        if not energy_cons or energy_cons.strip() == "":
            st.error("Please enter the target (Dependent) column name.")
        # 2) Validate all independent variable names are provided (not blank)
        elif any(v == "" for v in ind_vars):
            missing_idx = [str(i + 1) for i, v in enumerate(ind_vars) if v == ""]
            st.error(f"Please provide names for Independent Variable(s): {', '.join(missing_idx)}.")
        else:
            # 3) Validate these columns exist in uploaded df
            required_cols = [energy_cons] + ind_vars
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                st.error(f"The following column(s) are missing from the uploaded CSV: {missing_cols}")
            else:
                # 4) Convert to numeric where needed and handle errors
                try:
                    X = df[ind_vars].apply(pd.to_numeric)
                    y = pd.to_numeric(df[energy_cons])
                except Exception as e:
                    st.error(
                        f"Non-numeric data found in the selected columns. Convert to numeric or choose other columns. ({e})")
                else:
                    # Train/test + metrics
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    ModelClass = model_dict[model_choice]
                    model = ModelClass()
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    # metrics: R^2, RMSE, CVRMSE
                    from sklearn.metrics import mean_squared_error
                    import numpy as np

                    r2 = model.score(X_test, y_test)
                    rmse = mean_squared_error(y_test, preds, squared=False)
                    cvrmse = rmse / y_test.mean() if y_test.mean() != 0 else float("nan")

                    st.write(f'Regression (R²): {r2:.4f}')
                    st.write(f'RMSE: {rmse:.4f}')
                    st.write(f'CVRMSE: {cvrmse:.2%}')
                    st.line_chart(
                        pd.DataFrame({'Actual': y_test.reset_index(drop=True), 'Predicted': preds}).reset_index(
                            drop=True))


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