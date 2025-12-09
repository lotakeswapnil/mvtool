# mvtool.py
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from weather import make_openmeteo_client, fetch_openmeteo_archive
from datetime import date

st.title('Energy M&V Tool')


# --- Create session-state variable ---
if "mode" not in st.session_state:
    st.session_state.mode = None


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

        data_dict = {'Temperature', 'Independent Variable'}
        data_ind_var = st.selectbox('Select Independent Variable Type', data_dict)

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


    # --- Create session-state variable for Manual Mode---
    if "yes_no" not in st.session_state:
        st.session_state.yes_no = None

    # --- Display Start Buttons ---
    if st.session_state.yes_no is None:
        st.markdown('Do you want Weather Data?')

        col1, col2 = st.columns([0.05, 0.5])

        with col1:
            if st.button('Yes'):
                st.session_state.yes_no = 'yes'
                st.rerun()

        with col2:
            if st.button('No'):
                st.session_state.yes_no = 'no'
                st.rerun()


    if st.session_state.yes_no == 'no':

        if st.button("Back to Weather Data selection"):
            st.session_state.yes_no = None
            st.rerun()


        # --- Create session-state variable for Manual Mode---
        if "manual_data" not in st.session_state:
            st.session_state.manual_data = None

        # --- Display Start Buttons ---
        if st.session_state.manual_data is None:
            st.markdown('Does you data include Temperature or an Independent Variable?')

            temp1, temp2 = st.columns([0.25, 0.5])

            with temp1:
                if st.button('Temperature'):
                    st.session_state.manual_data = 'temp'
                    st.rerun()

            with temp2:
                if st.button('Independent Variable'):
                    st.session_state.manual_data = 'ind'
                    st.rerun()

        if st.session_state.manual_data == 'ind':

            if st.button("Back to Temperature or Variable selection"):
                st.session_state.manual_data = None
                st.rerun()

            # Ask for number of rows & columns
            num_cols = st.number_input("Number of Independent Variables: ", 0, 10, 3)

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
                final_df = st.data_editor(df_empty, num_rows="dynamic")
                st.dataframe(final_df)

            else:
                st.info('Please complete all Independent Variable names.')

        if st.session_state.manual_data == 'temp':

            # Build column names automatically
            col_names = ["Dependent Variable","Temperature"]  # first column fixed
            empty_df = pd.DataFrame("", index=range(0), columns=col_names)

            st.write('#### Enter Dependent Variable Below:')

            manual_df = st.data_editor(empty_df, num_rows="dynamic")



    elif st.session_state.yes_no == 'yes':

        if st.button("Back to Weather Data selection"):
            st.session_state.yes_no = None
            st.rerun()


        # Build column names automatically
        col_names = ["Dependent Variable"]  # first column fixed
        empty_df = pd.DataFrame("", index=range(0), columns=col_names)

        st.write('#### Enter Dependent Variable Below:')

        manual_df = st.data_editor(empty_df, num_rows="dynamic")

        # Fetch Weather Data
        interval_dict = {'Hourly', 'Daily', 'Monthly'}
        weather_interval = st.selectbox('Select Interval', interval_dict)

        lat = st.number_input("Latitude", format="%.4f")
        lon = st.number_input("Longitude", format="%.4f")
        start_date = st.date_input("Start date", value=date.today().replace(year=date.today().year-1).replace(day=date.today().day-1))
        end_date = st.date_input("End date", value=date.today().replace(day=date.today().day-2))
        var = "temperature"   # or let user pick
        which = "hourly"

        # create client once (you can cache it)
        @st.cache_resource
        def get_client():
            return make_openmeteo_client()

        client = get_client()

        if st.button("Fetch Weather Data"):
            if start_date > end_date:
                st.error("Start must be <= end")
            else:
                start_str = start_date.isoformat()
                end_str = end_date.isoformat()
                with st.spinner("Fetching..."):
                    try:
                        meta, df_weather = fetch_openmeteo_archive(client, lat, lon, start_str, end_str, which, var)
                    except Exception as e:
                        st.error(f"Weather fetch failed: {e}")
                    else:
                        st.success("Fetched weather data")
                        #st.json(meta)
                        #st.write(df_weather)

                        df = df_weather.copy()

                        if weather_interval == "Hourly":

                            df_weather_final = df

                        elif weather_interval == "Daily":

                            # Create a daily period column
                            df["daily"] = df["date_local"].dt.to_period("D")

                            # Group by day and compute averages
                            df_weather_final = (df.groupby("daily").mean(numeric_only=True).reset_index())

                        else:

                            # Create a monthly period column
                            df["month"] = df["date_local"].dt.to_period("M")

                            # Group by month and compute averages
                            df_weather_final = (df.groupby("month").mean(numeric_only=True).reset_index())


                        st.subheader("Combined Dependent Variable and Weather Data")
                        final_df = pd.concat([manual_df, df_weather_final], axis=1)
                        st.dataframe(final_df)










