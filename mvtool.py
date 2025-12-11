# import necessary packages
from datetime import date

import pandas as pd
import streamlit as st
from supplementary.change_point import (fit_three_param_cp, fit_five_param_deadband, select_model_by_rmse_r2,
                                        predict_3p_for_plot, predict_5p_for_plot)
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

from supplementary.weather import make_openmeteo_client, fetch_openmeteo_archive

st.title('Energy M&V Tool')

st.sidebar.header("Menu")

# --- Create session-state variable ---
if "mode" not in st.session_state:
    st.session_state.mode = None


# --- Display Start Buttons ---

if st.session_state.mode is None:
    st.subheader('Select Any One of the Options')

if st.session_state.mode is None:
    man, up = st.columns(2)

    with man:
        if st.button("Enter Data (Manual)"):
            st.session_state.mode = "manual"
            st.rerun()

    with up:
        if st.button("Upload Data (CSV)"):
            st.session_state.mode = "upload"
            st.rerun()

# -------------------------
# UPLOAD DATA MODE
# -------------------------

elif st.session_state.mode == "upload":


    if st.sidebar.button("Back to Main Menu"):
        st.session_state.mode = None
        st.rerun()

    st.subheader('Upload Data (CSV)')
    uploaded = st.file_uploader('', type="csv", label_visibility='collapsed')

    if uploaded:

        df = pd.read_csv(uploaded)
        st.write('### Preview:', df.head())

        data_ind_var = st.selectbox('Select Independent Variable Type', {'Temperature', 'Independent Variable'})


        if data_ind_var == 'Independent Variable':

            # Target (dependent) column
            energy_cons = st.text_input('Energy Consumption column name')

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
                        # ---------- ADDED: build list of independent variables ----------
                        independent = [
                            globals()[f"ind_var_{j}"]
                            for j in range(1, num_var + 1)
                            if globals()[f"ind_var_{j}"] in df.columns
                        ]
                        # -----------------------------------------------------------------

                        # ---------- MODIFIED MINIMALLY: use the list instead of single var ----------
                        X = df[independent]  # <- works for 1 or many variables
                        # ------------------------------------------------------------------------------

                        y = df[energy_cons]

                        model = model_dict[model_list]()
                        model.fit(X, y)
                        preds = model.predict(X)
                        regression = model.score(X, y)
                        cvrmse = root_mean_squared_error(y, preds)/y.mean()

                        # ---------- ADDED: Regression Equation Display ----------
                        coef = model.coef_
                        intercept = model.intercept_

                        equation_latex = (
                                "Energy = " +
                                f"{intercept:.4f} + " +
                                " + ".join([f"{coef[k]:.4f} \\times {independent[k]}" for k in range(len(independent))])
                        )

                        st.subheader("Regression Equation")
                        st.latex(equation_latex)

                        st.write(f'Regression: {regression:.2%}')
                        st.write(f'CVRMSE: {cvrmse:.2%}')
                        st.line_chart(pd.DataFrame({'Actual': y, 'Predicted': preds}).reset_index(drop=True))



                else:
                    st.error('All variables not defined.')

        # -------------------------
        # Sample data (heating + deadband + cooling)
        # -------------------------

        if data_ind_var == 'Temperature':

            temp_data = st.text_input('Temperature column name')
            energy_data = st.text_input('Energy column name')

            if temp_data != '' and energy_data != '':

                # -------------------------
                # DEFAULT MODEL SETTINGS
                # -------------------------
                Tmin = float(np.floor(df[temp_data].min()))
                Tmax = float(np.ceil(df[temp_data].max()))
                step = 1.0
                rel_tol_pct = 0.1  # 0.1% RMSE tie tolerance

                # -------------------------
                # RUN MODELS
                # -------------------------
                temp = df[temp_data].values
                kwh = df[energy_data].values

                with st.spinner("Running change-point models..."):
                    three_res = fit_three_param_cp(temp, kwh, Tmin, Tmax, step)
                    five_res = fit_five_param_deadband(temp, kwh, Tmin, Tmax, step)

                mean_kwh = float(df[energy_data].mean())
                preferred_label, preferred_result = select_model_by_rmse_r2(
                    three_res, five_res, rel_tol_pct, mean_kwh
                )

                # -------------------------
                # EQUATION DISPLAY
                # -------------------------
                st.write("## Model Equations")

                st.write('### 3-parameter:')
                st.latex(fr"\text{{Energy}} = {three_res['model'].intercept_:.2f} + {three_res['model'].coef_[0]:.2f}\,\max(0,\,T - {three_res['Tb']:.2f})")

                st.write('### 5-parameter:')
                st.latex(fr"\text{{Energy}} = {five_res['model'].intercept_:.2f} + {five_res['model'].coef_[0]:.2f}\,\max(0,\,{five_res['Tb_low']:.2f} - T) + {five_res['model'].coef_[1]:.2f}\,\max(0,\,T - {five_res['Tb_high']:.2f})")


                # -------------------------
                # DISPLAY RESULTS
                # -------------------------
                st.write("## Model Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("3-Parameter Model")
                    st.write(f"**Tb:** {three_res['Tb']:.2f} °C")
                    st.write(f"**β0:** {three_res['model'].intercept_:.2f}")
                    st.write(f"**β1:** {three_res['model'].coef_[0]:.2f}")
                    st.write(f"**RMSE:** {three_res['rmse']:.2f}")
                    st.write(f"**R²:** {three_res['r2']:.2f}")

                with col2:
                    st.subheader("5-Parameter Model")
                    st.write(f"**Tb_low:** {five_res['Tb_low']:.2f} °C")
                    st.write(f"**Tb_high:** {five_res['Tb_high']:.2f} °C")
                    st.write(f"**β0:** {five_res['model'].intercept_:.2f}")
                    st.write(f"**β_h:** {five_res['model'].coef_[0]:.2f}")
                    st.write(f"**β_c:** {five_res['model'].coef_[1]:.2f}")
                    st.write(f"**RMSE:** {five_res['rmse']:.2f}")
                    st.write(f"**R²:** {five_res['r2']:.2f}")

                st.success(f"### Preferred model → **{preferred_label}**")

                # -------------------------
                # PLOT MODELS
                # -------------------------
                T_plot = np.linspace(df[temp_data].min(), df[temp_data].max(), 400)

                Y3_plot = predict_3p_for_plot(T_plot, three_res["Tb"], three_res["model"])
                Y5_plot = predict_5p_for_plot(T_plot, five_res["Tb_low"], five_res["Tb_high"], five_res["model"])

                fig, ax = plt.subplots(figsize=(9, 5))
                ax.scatter(df[temp_data], df[energy_data], label="Measured Energy", s=50)

                if preferred_label == "3-parameter":
                    ax.plot(T_plot, Y3_plot, label="3-parameter (preferred)", linewidth=2.5)
                    ax.plot(T_plot, Y5_plot, '--', label="5-parameter", alpha=0.8)
                else:
                    ax.plot(T_plot, Y3_plot, '--', label="3-parameter", alpha=0.8)
                    ax.plot(T_plot, Y5_plot, label="5-parameter (preferred)", linewidth=2.5)

                # Deadband shade
                ax.axvspan(five_res["Tb_low"], five_res["Tb_high"], alpha=0.08, color="gray", label="Deadband")

                ax.set_xlabel("Temperature (°C)")
                ax.set_ylabel("Energy")
                ax.set_title("3-Parameter vs 5-Parameter Change-Point Models")
                ax.legend()
                ax.grid(True)

                st.pyplot(fig)

            if temp_data == '':
                st.error("Please add temperature column name.")

            if energy_data == '':
                st.error("Please add energy column name.")



# -------------------------
# MANUAL DATA MODE
# -------------------------

elif st.session_state.mode == "manual":

    if st.sidebar.button("Back to Menu"):
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

        if st.sidebar.button("Back to Weather Data selection"):
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

            if st.sidebar.button("Back to Temperature or Variable selection"):
                st.session_state.manual_data = None
                st.rerun()

            # Ask for number of rows & columns
            num_cols = st.number_input("Number of Independent Variables: ", 0, 10, 1)

            # Build column names automatically
            col_names = ["Energy"]  # first column fixed
            input_valid = True  # flag to track if all names are filled

            # Generate independent variable labels
            independent_vars = []

            # Generate independent variable labels
            for i in range(1, num_cols + 1):
                independent = st.text_input(f'Independent Variable {i}:', key=f"ind_var_{i}")

                # If blank, trigger error and mark input as invalid
                if independent.strip() == "":
                    st.error(f'Independent Variable {i} cannot be blank.')
                    input_valid = False

                col_names.append(independent)
                independent_vars.append(independent)

            # Only proceed if all variable names are valid
            if input_valid:
                df_empty = pd.DataFrame({"Energy": pd.Series([0], dtype=float),**{col: pd.Series([0.0], dtype=float) for col in col_names}})

                st.subheader('Enter Data Below:')
                final_df = st.data_editor(df_empty, num_rows="dynamic")

                model_dict = {'Linear Regression': LinearRegression, 'Ridge Regression': Ridge,
                              'Lasso Regression': Lasso}
                model_list = st.selectbox('Select models', model_dict)

                if st.button('Run Regression'):
                    X = final_df[independent_vars]
                    y = final_df['Energy']

                    model = model_dict[model_list]()
                    model.fit(X, y)
                    preds = model.predict(X)
                    regression = model.score(X, y)
                    cvrmse = root_mean_squared_error(y, preds) / y.mean()

                    # ---------- ADDED: Regression Equation Output ----------
                    coef = model.coef_
                    intercept = model.intercept_

                    equation_latex = (
                            "Energy = "
                            f"{intercept:.4f} + "
                            + " + ".join([
                        f"{coef[i]:.4f} \\times {independent_vars[i]}"
                        for i in range(len(independent_vars))
                    ])
                    )

                    st.subheader("Regression Equation")
                    st.latex(equation_latex)

                    st.write(f'Regression: {regression:.2%}')
                    st.write(f'CVRMSE: {cvrmse:.2%}')
                    st.line_chart(pd.DataFrame({'Actual': y, 'Predicted': preds}).reset_index(drop=True))


            else:
                st.info('Please complete all Independent Variable names.')


        if st.session_state.manual_data == 'temp':

            if st.sidebar.button("Back to Temperature or Variable selection"):
                st.session_state.manual_data = None
                st.rerun()

            # Build column names automatically

            empty_df = pd.DataFrame({"Energy": pd.Series([0], dtype=float),"Temperature": pd.Series([0], dtype=float)})


            st.write('#### Enter Energy Data Below:')

            final_df = st.data_editor(empty_df, num_rows='dynamic')

            # -------------------------
            # VALIDATE USER INPUT
            # -------------------------

            # Check for empty DataFrame
            if len(final_df) == 0 or final_df.isna().all().all():
                st.error("Please enter at least one row of Energy and Temperature data.")
                st.stop()

            # Check for missing values
            if final_df.isna().any().any():
                st.error("Some cells are empty. Please fill in all Energy and Temperature values.")
                st.stop()

            # Check for numeric values
            try:
                final_df = final_df.astype(float)
            except ValueError:
                st.error("All values must be numeric. Please correct invalid entries.")
                st.stop()

            # Check minimum dataset size for regression
            if len(final_df) < 2:
                st.error("At least two data points are required to fit a model.")
                st.stop()

            # -------------------------
            # DEFAULT MODEL SETTINGS
            # -------------------------

            Tmin = st.number_input('Lower Temperature', min = final_df['Temperature'].min().values(), max = final_df['Temperature'].max().values)
            Tmax = float(np.ceil(final_df['Temperature'].max()))

            step = 1.0
            rel_tol_pct = 0.1  # 0.1% RMSE tie tolerance


            # -------------------------
            # RUN MODELS
            # -------------------------
            temp = final_df['Temperature'].values
            kwh = final_df['Energy'].values

            model_choice = st.selectbox("Select Change-Point Model:",["3-parameter", "5-parameter", "Both"])

            with st.spinner("Running change-point models..."):
                three_res = None
                five_res = None

                if model_choice in ["3-parameter", "Both"]:
                    three_res = fit_three_param_cp(temp, kwh, Tmin, Tmax, step)

                if model_choice in ["5-parameter", "Both"]:
                    five_res = fit_five_param_deadband(temp, kwh, Tmin, Tmax, step)



            mean_kwh = float(final_df['Energy'].mean())
            #preferred_label, preferred_result = select_model_by_rmse_r2(three_res, five_res, rel_tol_pct, mean_kwh)

            # -------------------------
            # EQUATION DISPLAY
            # -------------------------

            st.write("## Model Equations")

            if model_choice in ["3-parameter", "Both"]:
                st.write('### 3-parameter:')
                st.latex(
                    fr"\text{{Energy}} = {three_res['model'].intercept_:.2f} + "
                    fr"{three_res['model'].coef_[0]:.2f}\,\max(0,\,T - {three_res['Tb']:.2f})"
                )

            if model_choice in ["5-parameter", "Both"]:
                st.write('### 5-parameter:')
                st.latex(
                    fr"\text{{Energy}} = {five_res['model'].intercept_:.2f} + "
                    fr"{five_res['model'].coef_[0]:.2f}\,\max(0,\,{five_res['Tb_low']:.2f} - T) + "
                    fr"{five_res['model'].coef_[1]:.2f}\,\max(0,\,T - {five_res['Tb_high']:.2f})"
                )

            # -------------------------
            # DISPLAY RESULTS
            # -------------------------
            st.write("## Model Results")

            if model_choice in ["3-parameter"]:
                st.subheader("3-Parameter Model")
                st.write(f"**Tb:** {three_res['Tb']:.2f} °C")
                st.write(f"**β0:** {three_res['model'].intercept_:.2f}")
                st.write(f"**β1:** {three_res['model'].coef_[0]:.2f}")
                st.write(f"**RMSE:** {three_res['rmse']:.2f}")
                st.write(f"**R²:** {three_res['r2']:.2f}")

            if model_choice in ["5-parameter"]:
                st.subheader("5-Parameter Model")
                st.write(f"**Tb_low:** {five_res['Tb_low']:.2f} °C")
                st.write(f"**Tb_high:** {five_res['Tb_high']:.2f} °C")
                st.write(f"**β0:** {five_res['model'].intercept_:.2f}")
                st.write(f"**β_h:** {five_res['model'].coef_[0]:.2f}")
                st.write(f"**β_c:** {five_res['model'].coef_[1]:.2f}")
                st.write(f"**RMSE:** {five_res['rmse']:.2f}")
                st.write(f"**R²:** {five_res['r2']:.2f}")

            if model_choice in ["Both"]:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("3-Parameter Model")
                    st.write(f"**Tb:** {three_res['Tb']:.2f} °C")
                    st.write(f"**β0:** {three_res['model'].intercept_:.2f}")
                    st.write(f"**β1:** {three_res['model'].coef_[0]:.2f}")
                    st.write(f"**RMSE:** {three_res['rmse']:.2f}")
                    st.write(f"**R²:** {three_res['r2']:.2f}")
                with col2:
                    st.subheader("5-Parameter Model")
                    st.write(f"**Tb_low:** {five_res['Tb_low']:.2f} °C")
                    st.write(f"**Tb_high:** {five_res['Tb_high']:.2f} °C")
                    st.write(f"**β0:** {five_res['model'].intercept_:.2f}")
                    st.write(f"**β_h:** {five_res['model'].coef_[0]:.2f}")
                    st.write(f"**β_c:** {five_res['model'].coef_[1]:.2f}")
                    st.write(f"**RMSE:** {five_res['rmse']:.2f}")
                    st.write(f"**R²:** {five_res['r2']:.2f}")



            # -------------------------
            # PLOT MODELS
            # -------------------------
            T_plot = np.linspace(final_df['Temperature'].min(), final_df['Temperature'].max(), 400)


            fig, ax = plt.subplots(figsize=(9, 5))
            ax.scatter(final_df['Temperature'], final_df['Energy'], label="Measured Energy", s=50)

            if model_choice == "3-parameter":
                Y3_plot = predict_3p_for_plot(T_plot, three_res["Tb"], three_res["model"])
                ax.plot(T_plot, Y3_plot, label="3-parameter", linewidth=2.5)

            elif model_choice == "5-parameter":
                Y5_plot = predict_5p_for_plot(T_plot, five_res["Tb_low"], five_res["Tb_high"], five_res["model"])
                ax.plot(T_plot, Y5_plot, label="5-parameter", linewidth=2.5)

            else:  # Both
                Y3_plot = predict_3p_for_plot(T_plot, three_res["Tb"], three_res["model"])
                Y5_plot = predict_5p_for_plot(T_plot, five_res["Tb_low"], five_res["Tb_high"], five_res["model"])
                ax.plot(T_plot, Y3_plot, label="3-parameter", linewidth=2.5)
                ax.plot(T_plot, Y5_plot, label="5-parameter", linewidth=2.5)

            # Deadband shade
            if model_choice in ["5-parameter", "Both"]:
                ax.axvspan(five_res["Tb_low"], five_res["Tb_high"], alpha=0.08, color="gray", label="Deadband")

            ax.set_xlabel("Temperature (°C)")
            ax.set_ylabel("Energy")
            ax.set_title("3-Parameter vs 5-Parameter Change-Point Models")
            ax.legend()
            ax.grid(True)

            st.pyplot(fig)



    elif st.session_state.yes_no == 'yes':

        if st.sidebar.button("Back to Weather Data selection"):
            st.session_state.yes_no = None
            st.rerun()


        # Build column names automatically
        col_names = ["Energy"]  # first column fixed
        empty_df = pd.DataFrame({"Energy": pd.Series([0], dtype=float)})

        st.write('#### Enter Energy Data Below:')

        manual_df = st.data_editor(empty_df, num_rows="dynamic")

        if len(manual_df) < 2:
            st.error("Please enter at least 2 rows.")


        # Fetch Weather Data


        lat, lon = st.columns(2)

        with lat:
            lat = st.number_input("Latitude", format="%.4f")
            start_date = st.date_input("Start date", value=date.today().replace(year=date.today().year - 1).replace(
                day=date.today().day - 1))

        with lon:
            lon = st.number_input("Longitude", format="%.4f")
            end_date = st.date_input("End date", value=date.today().replace(day=date.today().day - 2))

        var = "temperature"   # or let user pick
        which = "hourly"

        # create client once (you can cache it)
        @st.cache_resource
        def get_client():
            return make_openmeteo_client()

        client = get_client()

        weather_i,model_c = st.columns(2)

        with weather_i:
            weather_interval = st.selectbox('Select Interval', {'Hourly', 'Daily', 'Monthly'})

        with model_c:
            model_choice = st.selectbox("Select Change-Point Model:", ["3-parameter", "5-parameter", "Both"])

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

                        elif weather_interval == "Monthly":

                            df["month"] = df["date_local"].dt.month  # 1–12

                            df_weather_final = (df.groupby("month", as_index=False).mean(numeric_only=True))

                        else:

                            df["month"] = df["date_local"].dt.month
                            df["day"] = df["date_local"].dt.day

                            df_weather_final = (df.groupby(["month", "day"], as_index=False).mean(numeric_only=True))

                        st.subheader("Combined Dependent Variable and Weather Data")
                        final_df = pd.concat([manual_df, df_weather_final], axis=1)
                        st.dataframe(final_df)

                        # -------------------------
                        # DEFAULT MODEL SETTINGS
                        # -------------------------

                        Tmin = float(np.floor(final_df['temperature'].min()))
                        Tmax = float(np.ceil(final_df['temperature'].max()))
                        step = 1.0
                        rel_tol_pct = 0.1  # 0.1% RMSE tie tolerance


                        # -------------------------
                        # RUN MODELS
                        # -------------------------
                        temp = final_df['temperature'].values
                        kwh = final_df['Energy'].values

                        with st.spinner("Running change-point models..."):
                            three_res = None
                            five_res = None

                            if model_choice in ["3-parameter", "Both"]:
                                three_res = fit_three_param_cp(temp, kwh, Tmin, Tmax, step)

                            if model_choice in ["5-parameter", "Both"]:
                                five_res = fit_five_param_deadband(temp, kwh, Tmin, Tmax, step)

                        mean_kwh = float(final_df['Energy'].mean())
                        #preferred_label, preferred_result = select_model_by_rmse_r2(three_res, five_res, rel_tol_pct,mean_kwh)

                        # -------------------------
                        # EQUATION DISPLAY
                        # -------------------------
                        st.write("## Model Equations")

                        if model_choice in ["3-parameter", "Both"]:
                            st.write('### 3-parameter:')
                            st.latex(
                                    fr"\text{{Energy}} = {three_res['model'].intercept_:.2f} + "
                                    fr"{three_res['model'].coef_[0]:.2f}\,\max(0,\,T - {three_res['Tb']:.2f})"
                                    )

                        if model_choice in ["5-parameter", "Both"]:
                            st.write('### 5-parameter:')
                            st.latex(
                                    fr"\text{{Energy}} = {five_res['model'].intercept_:.2f} + "
                                    fr"{five_res['model'].coef_[0]:.2f}\,\max(0,\,{five_res['Tb_low']:.2f} - T) + "
                                    fr"{five_res['model'].coef_[1]:.2f}\,\max(0,\,T - {five_res['Tb_high']:.2f})"
                                    )

                        # -------------------------
                        # DISPLAY RESULTS
                        # -------------------------
                        st.write("## Model Results")

                        if model_choice in ["3-parameter"]:
                            st.subheader("3-Parameter Model")
                            st.write(f"**Tb:** {three_res['Tb']:.2f} °C")
                            st.write(f"**β0:** {three_res['model'].intercept_:.2f}")
                            st.write(f"**β1:** {three_res['model'].coef_[0]:.2f}")
                            st.write(f"**RMSE:** {three_res['rmse']:.2f}")
                            st.write(f"**R²:** {three_res['r2']:.2f}")

                        if model_choice in ["5-parameter"]:
                            st.subheader("5-Parameter Model")
                            st.write(f"**Tb_low:** {five_res['Tb_low']:.2f} °C")
                            st.write(f"**Tb_high:** {five_res['Tb_high']:.2f} °C")
                            st.write(f"**β0:** {five_res['model'].intercept_:.2f}")
                            st.write(f"**β_h:** {five_res['model'].coef_[0]:.2f}")
                            st.write(f"**β_c:** {five_res['model'].coef_[1]:.2f}")
                            st.write(f"**RMSE:** {five_res['rmse']:.2f}")
                            st.write(f"**R²:** {five_res['r2']:.2f}")

                        if model_choice in ["Both"]:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("3-Parameter Model")
                                st.write(f"**Tb:** {three_res['Tb']:.2f} °C")
                                st.write(f"**β0:** {three_res['model'].intercept_:.2f}")
                                st.write(f"**β1:** {three_res['model'].coef_[0]:.2f}")
                                st.write(f"**RMSE:** {three_res['rmse']:.2f}")
                                st.write(f"**R²:** {three_res['r2']:.2f}")
                            with col2:
                                st.subheader("5-Parameter Model")
                                st.write(f"**Tb_low:** {five_res['Tb_low']:.2f} °C")
                                st.write(f"**Tb_high:** {five_res['Tb_high']:.2f} °C")
                                st.write(f"**β0:** {five_res['model'].intercept_:.2f}")
                                st.write(f"**β_h:** {five_res['model'].coef_[0]:.2f}")
                                st.write(f"**β_c:** {five_res['model'].coef_[1]:.2f}")
                                st.write(f"**RMSE:** {five_res['rmse']:.2f}")
                                st.write(f"**R²:** {five_res['r2']:.2f}")


                        # -------------------------
                        # PLOT MODELS
                        # -------------------------
                        T_plot = np.linspace(final_df['temperature'].min(), final_df['temperature'].max(), 400)

                        fig, ax = plt.subplots(figsize=(9, 5))
                        ax.scatter(final_df['temperature'], final_df['Energy'], label="Measured Energy", s=50)

                        if model_choice == "3-parameter":
                            Y3_plot = predict_3p_for_plot(T_plot, three_res["Tb"], three_res["model"])
                            ax.plot(T_plot, Y3_plot, label="3-parameter", linewidth=2.5)

                        elif model_choice == "5-parameter":
                            Y5_plot = predict_5p_for_plot(T_plot, five_res["Tb_low"], five_res["Tb_high"], five_res["model"])
                            ax.plot(T_plot, Y5_plot, label="5-parameter", linewidth=2.5)

                        else:  # Both
                            Y3_plot = predict_3p_for_plot(T_plot, three_res["Tb"], three_res["model"])
                            Y5_plot = predict_5p_for_plot(T_plot, five_res["Tb_low"], five_res["Tb_high"], five_res["model"])
                            ax.plot(T_plot, Y3_plot, label="3-parameter", linewidth=2.5)
                            ax.plot(T_plot, Y5_plot, label="5-parameter", linewidth=2.5)


                        # Deadband shade
                        if model_choice in ["5-parameter", "Both"]:
                            ax.axvspan(five_res["Tb_low"], five_res["Tb_high"], alpha=0.08, color="gray", label="Deadband")

                        ax.set_xlabel("Temperature (°C)")
                        ax.set_ylabel("Energy")
                        ax.set_title("3-Parameter vs 5-Parameter Change-Point Models")
                        ax.legend()
                        ax.grid(True)

                        st.pyplot(fig)
