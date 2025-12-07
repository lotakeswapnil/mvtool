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
    st.subheader('Upload Data (CSV)')
    uploaded = st.file_uploader(type="csv")

    if st.button("Back to Menu"):
        st.session_state.mode = None
        st.rerun()

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write('Preview:', df.head())

        energy_cons = st.text_input('Target column name (energy usage)')
        num_var = st.number_input('Number of variables', min_value=1, max_value=10, step=1)
        model_dict = {'Linear Regression':LinearRegression, 'Ridge Regression':Ridge, 'Lasso Regression':Lasso}
        model_list = st.selectbox('Select models', model_dict)


        for i in range(1,num_var+1):
            globals()[f"ind_var_{i}"] = st.text_input(f"Independent Variable {i}",key=f"var_{i}")

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

    # Ask for number of rows & columns
    num_cols = st.number_input("Number of columns", 1, 10, 3)

    # Ask for column names
    st.subheader("Enter Column Names")
    col_names = []
    for i in range(num_cols):
        col = st.text_input(f"Column {i + 1} name", key=f"col_{i}")
        col_names.append(col)

    # Only show table if all column names are provided
    if all(col_names):

        # Create empty dataframe
        df_empty = pd.DataFrame("", index=range(1), columns=col_names)

        st.subheader("Enter Data Below:")
        edited_df = st.data_editor(df_empty, num_rows="dynamic")

        if st.button("Create DataFrame"):
            st.success("Generated DataFrame:")
            st.dataframe(edited_df)

    else:
        st.info("Please enter all column names to show the input table.")