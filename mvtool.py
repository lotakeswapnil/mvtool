# mvtool.py
from datetime import date

import pandas as pd
import streamlit as st
from change_point import (fit_three_param_cp, fit_five_param_deadband, select_model_by_rmse_r2,
                                predict_3p_for_plot, predict_5p_for_plot)
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from weather import make_openmeteo_client, fetch_openmeteo_archive

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

        data_ind_var = st.selectbox('Select Independent Variable Type', {'Temperature', 'Independent Variable'})


        if data_ind_var == 'Independent Variable':

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
        # Sample data (heating + deadband + cooling)
        # -------------------------

        if data_ind_var == 'Temperature':

            st.write("### Sample data")
            st.dataframe(df)

            temp_data = st.text_input('Temperature column name')
            energy_data = st.text_input('Energy column name')


            if temp_data != '' and energy_data != '':
                # Sidebar settings
                st.sidebar.header("Model settings")
                Tmin = st.sidebar.number_input("Search Tmin (°C)", value=float(np.floor(df[temp_data].min())))
                Tmax = st.sidebar.number_input("Search Tmax (°C)", value=float(np.ceil(df[temp_data].max())))
                step = st.sidebar.number_input("Search step (°C)", value=1.0, step=0.5)
                rel_tol_pct = st.sidebar.slider("RMSE tie tolerance (%)", min_value=0.0, max_value=5.0, value=0.1, step=0.1)
                run_button = st.sidebar.button("Run models")

                # Always run (or use run_button if you prefer explicit trigger)
                if run_button or True:
                    temp = df[temp_data].values
                    energy = df[energy_data].values

                    with st.spinner("Fitting models..."):
                        three_res = fit_three_param_cp(temp, energy, Tmin=Tmin, Tmax=Tmax, step=step)
                        five_res = fit_five_param_deadband(temp, energy, Tmin=Tmin, Tmax=Tmax, step=step)

                    mean_kwh = float(df[energy_data].mean())
                    preferred_label, preferred_result = select_model_by_rmse_r2(three_res, five_res, rel_tol_pct, mean_kwh)

                    # Present results (2 decimals)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("3-Parameter model")
                        st.write(f"**Tb (°C):** {three_res['Tb']:.2f}")
                        st.write(f"**β0:** {three_res['model'].intercept_:.2f}")
                        st.write(f"**β1:** {three_res['model'].coef_[0]:.2f}")
                        st.write(f"**RMSE:** {three_res['rmse']:.2f}")
                        st.write(f"**R²:** {three_res['r2']:.2f}")

                    with col2:
                        st.subheader("5-Parameter model")
                        st.write(f"**Tb_low (°C):** {five_res['Tb_low']:.2f}")
                        st.write(f"**Tb_high (°C):** {five_res['Tb_high']:.2f}")
                        st.write(f"**β0:** {five_res['model'].intercept_:.2f}")
                        st.write(f"**β_h:** {five_res['model'].coef_[0]:.2f}")
                        st.write(f"**β_c:** {five_res['model'].coef_[1]:.2f}")
                        st.write(f"**RMSE:** {five_res['rmse']:.2f}")
                        st.write(f"**R²:** {five_res['r2']:.2f}")

                    st.success(f"Preferred model (RMSE primary, R² tiebreaker): **{preferred_label}**")

                    # Add predictions columns to dataframe (both models)
                    df = df.copy()
                    # 3p preds
                    Tb = three_res["Tb"]
                    df["pred_3p"] = three_res["model"].predict(np.maximum(0.0, df[temp_data].values - Tb).reshape(-1, 1))
                    # 5p preds
                    Tb_low = five_res["Tb_low"];
                    Tb_high = five_res["Tb_high"]
                    heat = np.maximum(0.0, Tb_low - df[temp_data].values)
                    cool = np.maximum(0.0, df[temp_data].values - Tb_high)
                    df["pred_5p"] = five_res["model"].predict(np.column_stack([heat, cool]))

                    st.write("### Data with model predictions")
                    st.dataframe(
                        df.style.format({temp_data: "{:.2f}", energy_data: "{:.2f}", "pred_3p": "{:.2f}", "pred_5p": "{:.2f}"}))

                    # Plot measured points + both model curves
                    T_plot = np.linspace(df[temp_data].min(), df[temp_data].max(), 400)
                    Y3_plot = predict_3p_for_plot(T_plot, three_res["Tb"], three_res["model"])
                    Y5_plot = predict_5p_for_plot(T_plot, five_res["Tb_low"], five_res["Tb_high"], five_res["model"])

                    fig, ax = plt.subplots(figsize=(9, 5))
                    ax.scatter(df[temp_data], df[energy_data], label="Measured Energy", s=50, zorder=3)

                    # highlight preferred
                    if preferred_label == "3-parameter":
                        ax.plot(T_plot, Y3_plot, label="3-parameter (preferred)", linewidth=2.5)
                        ax.plot(T_plot, Y5_plot, label="5-parameter", linewidth=2, linestyle='--', alpha=0.8)
                    else:
                        ax.plot(T_plot, Y3_plot, label="3-parameter", linewidth=2, linestyle='--', alpha=0.8)
                        ax.plot(T_plot, Y5_plot, label="5-parameter (preferred)", linewidth=2.5)

                    # shade deadband region (5p)
                    ax.axvspan(five_res["Tb_low"], five_res["Tb_high"], alpha=0.08, color="grey", label="Deadband")

                    ax.set_xlabel("Temperature")
                    ax.set_ylabel("Energy")
                    ax.set_title("Measured Energy and model fits (3p vs 5p)")
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)

            if temp_data == '':
                st.error("Please add temperature column name.")

            if energy_data == '':
                st.error("Please add energy column name.")



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
            col_names = ["Energy"]  # first column fixed
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

            if st.button("Back to Temperature or Variable selection"):
                st.session_state.manual_data = None
                st.rerun()

            # Build column names automatically

            col_names = ["Energy","Temperature"]  # first column fixed
            empty_df = pd.DataFrame("", index=range(0), columns=col_names)


            st.write('#### Enter Energy Data Below:')

            final_df = st.data_editor(empty_df, num_rows='dynamic')

            for col in final_df.columns:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

            #final_df = st.dataframe(edited_df)

            # Sidebar settings
            st.sidebar.header("Model settings")
            Tmin = st.sidebar.number_input("Search Tmin (°C)",value=float(np.floor(final_df['Temperature'].min())))
            Tmax = st.sidebar.number_input("Search Tmax (°C)",value=float(np.ceil(final_df['Temperature'].max())))
            step = st.sidebar.number_input("Search step (°C)", value=1.0, step=0.5)
            rel_tol_pct = st.sidebar.slider("RMSE tie tolerance (%)", min_value=0.0, max_value=5.0,value=0.1, step=0.1)
            run_button = st.sidebar.button("Run models")

            # Always run (or use run_button if you prefer explicit trigger)
            if run_button or True:
                temp = final_df['Temperature'].values
                energy = final_df['Energy'].values

                with st.spinner("Fitting models..."):
                    three_res = fit_three_param_cp(temp, energy, Tmin=Tmin, Tmax=Tmax, step=step)
                    five_res = fit_five_param_deadband(temp, energy, Tmin=Tmin, Tmax=Tmax, step=step)

                mean_kwh = float(final_df['Energy'].mean())
                preferred_label, preferred_result = select_model_by_rmse_r2(three_res, five_res,
                                                                            rel_tol_pct, mean_kwh)

                # Present results (2 decimals)
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("3-Parameter model")
                    st.write(f"**Tb (°C):** {three_res['Tb']:.2f}")
                    st.write(f"**β0:** {three_res['model'].intercept_:.2f}")
                    st.write(f"**β1:** {three_res['model'].coef_[0]:.2f}")
                    st.write(f"**RMSE:** {three_res['rmse']:.2f}")
                    st.write(f"**R²:** {three_res['r2']:.2f}")

                with col2:
                    st.subheader("5-Parameter model")
                    st.write(f"**Tb_low (°C):** {five_res['Tb_low']:.2f}")
                    st.write(f"**Tb_high (°C):** {five_res['Tb_high']:.2f}")
                    st.write(f"**β0:** {five_res['model'].intercept_:.2f}")
                    st.write(f"**β_h:** {five_res['model'].coef_[0]:.2f}")
                    st.write(f"**β_c:** {five_res['model'].coef_[1]:.2f}")
                    st.write(f"**RMSE:** {five_res['rmse']:.2f}")
                    st.write(f"**R²:** {five_res['r2']:.2f}")

                st.success(f"Preferred model (RMSE primary, R² tiebreaker): **{preferred_label}**")

                # Add predictions columns to dataframe (both models)
                final_df = final_df.copy()
                # 3p preds
                Tb = three_res["Tb"]
                final_df["pred_3p"] = three_res["model"].predict(
                    np.maximum(0.0, final_df['Temperature'].values - Tb).reshape(-1, 1))
                # 5p preds
                Tb_low = five_res["Tb_low"];
                Tb_high = five_res["Tb_high"]
                heat = np.maximum(0.0, Tb_low - final_df['Temperature'].values)
                cool = np.maximum(0.0, final_df['Temperature'].values - Tb_high)
                final_df["pred_5p"] = five_res["model"].predict(np.column_stack([heat, cool]))

                st.write("### Data with model predictions")
                st.dataframe(
                    final_df.style.format({'Temperature': "{:.2f}", 'Energy': "{:.2f}", "pred_3p": "{:.2f}",
                                           "pred_5p": "{:.2f}"}))

                # Plot measured points + both model curves
                T_plot = np.linspace(final_df['Temperature'].min(), final_df['Temperature'].max(), 400)
                Y3_plot = predict_3p_for_plot(T_plot, three_res["Tb"], three_res["model"])
                Y5_plot = predict_5p_for_plot(T_plot, five_res["Tb_low"], five_res["Tb_high"],
                                              five_res["model"])

                fig, ax = plt.subplots(figsize=(9, 5))
                ax.scatter(final_df['Temperature'], final_df['Energy'], label="Measured Energy", s=50, zorder=3)

                # highlight preferred
                if preferred_label == "3-parameter":
                    ax.plot(T_plot, Y3_plot, label="3-parameter (preferred)", linewidth=2.5)
                    ax.plot(T_plot, Y5_plot, label="5-parameter", linewidth=2, linestyle='--',
                            alpha=0.8)
                else:
                    ax.plot(T_plot, Y3_plot, label="3-parameter", linewidth=2, linestyle='--',
                            alpha=0.8)
                    ax.plot(T_plot, Y5_plot, label="5-parameter (preferred)", linewidth=2.5)

                # shade deadband region (5p)
                ax.axvspan(five_res["Tb_low"], five_res["Tb_high"], alpha=0.08, color="grey",
                           label="Deadband")

                ax.set_xlabel("Temperature")
                ax.set_ylabel("Energy")
                ax.set_title("Measured Energy and model fits (3p vs 5p)")
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)



    elif st.session_state.yes_no == 'yes':

        if st.button("Back to Weather Data selection"):
            st.session_state.yes_no = None
            st.rerun()


        # Build column names automatically
        col_names = ["Energy"]  # first column fixed
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

                        temp_data = st.text_input('Temperature column name')
                        energy_data = st.text_input('Energy column name')

                        if temp_data != '' and energy_data != '':
                            # Sidebar settings
                            st.sidebar.header("Model settings")
                            Tmin = st.sidebar.number_input("Search Tmin (°C)",
                                                           value=float(np.floor(final_df[temp_data].min())))
                            Tmax = st.sidebar.number_input("Search Tmax (°C)",
                                                           value=float(np.ceil(final_df[temp_data].max())))
                            step = st.sidebar.number_input("Search step (°C)", value=1.0, step=0.5)
                            rel_tol_pct = st.sidebar.slider("RMSE tie tolerance (%)", min_value=0.0, max_value=5.0,
                                                            value=0.1, step=0.1)
                            run_button = st.sidebar.button("Run models")

                            # Always run (or use run_button if you prefer explicit trigger)
                            if run_button or True:
                                temp = final_df[temp_data].values
                                energy = final_df[energy_data].values

                                with st.spinner("Fitting models..."):
                                    three_res = fit_three_param_cp(temp, energy, Tmin=Tmin, Tmax=Tmax, step=step)
                                    five_res = fit_five_param_deadband(temp, energy, Tmin=Tmin, Tmax=Tmax, step=step)

                                mean_kwh = float(final_df[energy_data].mean())
                                preferred_label, preferred_result = select_model_by_rmse_r2(three_res, five_res,
                                                                                            rel_tol_pct, mean_kwh)

                                # Present results (2 decimals)
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.subheader("3-Parameter model")
                                    st.write(f"**Tb (°C):** {three_res['Tb']:.2f}")
                                    st.write(f"**β0:** {three_res['model'].intercept_:.2f}")
                                    st.write(f"**β1:** {three_res['model'].coef_[0]:.2f}")
                                    st.write(f"**RMSE:** {three_res['rmse']:.2f}")
                                    st.write(f"**R²:** {three_res['r2']:.2f}")

                                with col2:
                                    st.subheader("5-Parameter model")
                                    st.write(f"**Tb_low (°C):** {five_res['Tb_low']:.2f}")
                                    st.write(f"**Tb_high (°C):** {five_res['Tb_high']:.2f}")
                                    st.write(f"**β0:** {five_res['model'].intercept_:.2f}")
                                    st.write(f"**β_h:** {five_res['model'].coef_[0]:.2f}")
                                    st.write(f"**β_c:** {five_res['model'].coef_[1]:.2f}")
                                    st.write(f"**RMSE:** {five_res['rmse']:.2f}")
                                    st.write(f"**R²:** {five_res['r2']:.2f}")

                                st.success(f"Preferred model (RMSE primary, R² tiebreaker): **{preferred_label}**")

                                # Add predictions columns to dataframe (both models)
                                final_df = final_df.copy()
                                # 3p preds
                                Tb = three_res["Tb"]
                                final_df["pred_3p"] = three_res["model"].predict(
                                    np.maximum(0.0, final_df[temp_data].values - Tb).reshape(-1, 1))
                                # 5p preds
                                Tb_low = five_res["Tb_low"];
                                Tb_high = five_res["Tb_high"]
                                heat = np.maximum(0.0, Tb_low - final_df[temp_data].values)
                                cool = np.maximum(0.0, final_df[temp_data].values - Tb_high)
                                final_df["pred_5p"] = five_res["model"].predict(np.column_stack([heat, cool]))

                                st.write("### Data with model predictions")
                                st.dataframe(
                                    final_df.style.format({temp_data: "{:.2f}", energy_data: "{:.2f}", "pred_3p": "{:.2f}",
                                                     "pred_5p": "{:.2f}"}))

                                # Plot measured points + both model curves
                                T_plot = np.linspace(final_df[temp_data].min(), final_df[temp_data].max(), 400)
                                Y3_plot = predict_3p_for_plot(T_plot, three_res["Tb"], three_res["model"])
                                Y5_plot = predict_5p_for_plot(T_plot, five_res["Tb_low"], five_res["Tb_high"],
                                                              five_res["model"])

                                fig, ax = plt.subplots(figsize=(9, 5))
                                ax.scatter(final_df[temp_data], final_df[energy_data], label="Measured Energy", s=50, zorder=3)

                                # highlight preferred
                                if preferred_label == "3-parameter":
                                    ax.plot(T_plot, Y3_plot, label="3-parameter (preferred)", linewidth=2.5)
                                    ax.plot(T_plot, Y5_plot, label="5-parameter", linewidth=2, linestyle='--',
                                            alpha=0.8)
                                else:
                                    ax.plot(T_plot, Y3_plot, label="3-parameter", linewidth=2, linestyle='--',
                                            alpha=0.8)
                                    ax.plot(T_plot, Y5_plot, label="5-parameter (preferred)", linewidth=2.5)

                                # shade deadband region (5p)
                                ax.axvspan(five_res["Tb_low"], five_res["Tb_high"], alpha=0.08, color="grey",
                                           label="Deadband")

                                ax.set_xlabel("Temperature")
                                ax.set_ylabel("Energy")
                                ax.set_title("Measured Energy and model fits (3p vs 5p)")
                                ax.grid(True)
                                ax.legend()
                                st.pyplot(fig)

                        if temp_data == '':
                            st.error("Please add temperature column name.")

                        if energy_data == '':
                            st.error("Please add energy column name.")


