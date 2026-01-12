# --------------------------
# import necessary packages
# --------------------------

from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

from supplementary.change_point import (fit_three_param_cp, fit_five_param_deadband, predict_3p_for_plot,predict_5p_for_plot, model_details)
from supplementary.weather import make_openmeteo_client, fetch_openmeteo_archive, get_timezone_from_coords
from supplementary.model_results import three_para_results, five_para_results, three_five_para_results
from supplementary.weather import pvgis_tmy

# ----------------------------------------
# define manual data calculations function
# ----------------------------------------

def manual_page():

    st.subheader('Enter Data (Manual)')

    manual_data = st.radio('Do you want Weather Data?', options=["Yes", "No"], index=None, key='manual_data')

    if manual_data == 'No':

        # --- Display Start Buttons ---
        weather_data = st.radio('Does your data include Temperature?', options=["Yes", "No"], index=None, key='weather_data')

        if weather_data == 'No':

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

                st.subheader('Enter Baseline Data Below:')
                final_df = st.data_editor(df_empty, num_rows="dynamic", key='baseline_e')

                st.subheader('Enter Reported Data Below:')

                reported_df = st.data_editor(df_empty, num_rows="dynamic", key='reported_e')


                if st.button('Calculate Savings'):
                    X = final_df[independent_vars]
                    y = final_df['Energy']

                    model = LinearRegression()
                    model.fit(X, y)
                    preds = model.predict(X)
                    regression = model.score(X, y)
                    cvrmse = root_mean_squared_error(y, preds) / y.mean()

                    # ---------- ADDED: Regression Equation Output ----------
                    coef = model.coef_
                    intercept = model.intercept_

                    equation_latex = (
                            "Energy = "
                            f"{intercept:.2f} + "
                            + " + ".join([
                        f"{coef[i]:.2f} \\times {independent_vars[i]}"
                        for i in range(len(independent_vars))
                    ])
                    )

                    st.subheader("Regression Equation")
                    st.latex(equation_latex)

                    st.write(f'Regression: {regression:.2%}')
                    st.write(f'CVRMSE: {cvrmse:.2%}')
                    st.line_chart(pd.DataFrame({'Actual': y, 'Predicted': preds}).reset_index(drop=True))


                    X_r = reported_df[independent_vars]
                    Y_r = reported_df['Energy']
                    pred_r = model.predict(X_r)

                    st.write(f'##### Predicted Baseline Consumption: \n {pred_r.sum():.2f}')
                    st.write(f'##### Reported Consumption: \n {Y_r.sum():.2f}')
                    savings = pred_r.sum() - Y_r.sum()
                    st.write(f'##### Savings: \n {savings:.2f}')


            else:
                st.info('Please complete all Independent Variable names.')


        if weather_data == 'Yes':

            # Build column names automatically

            empty_df = pd.DataFrame({"Energy": pd.Series([0], dtype=float),"Temperature": pd.Series([0], dtype=float)})


            st.write('#### Enter Baseline Energy Data Below:')

            final_df = st.data_editor(empty_df, num_rows='dynamic', key='baseline_df')

            st.write('#### Enter Reported Energy Data Below:')

            reported_df = st.data_editor(empty_df, num_rows='dynamic', key='reported_df')

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

            Tmin = float(np.floor(final_df['Temperature'].min()))
            Tmax = float(np.ceil(final_df['Temperature'].max()))
            rel_tol_pct = 0.1  # 0.1% RMSE tie tolerance


            # -------------------------
            # RUN MODELS
            # -------------------------
            temp = final_df['Temperature'].values
            energy = final_df['Energy'].values

            temp_sel,cp_model = st.columns(2, border=True)

            with temp_sel:
                temperature_unit = st.selectbox('Select Temperature unit:',['celsius','fahrenheit'])
                model_choice = st.selectbox("Select Change-Point Model:", ["3-parameter", "5-parameter", "Both"])

            with cp_model:
                step = st.selectbox('Select Intervals For Balance Point:',[1.0,0.5,0.1], help='This interval will be used to calculate Balance Point.'
                                                                                             '\n E.g., If you select 1.0, balance point will be calculated between 55, 56, 57, and so on.'
                                                                                              '\n If you select 0.5, it will be calculated between 55.0, 55.5, 56.0, and so on.')
                if model_choice == "3-parameter":
                    mode = st.selectbox("Select Change-Point Model Type:",["auto", "heating", "cooling"],index=0)
                elif model_choice == "5-parameter":
                    # Disable the mode selection if the model is not "3-parameter"
                    mode_disabled = model_choice != "3-parameter"

                    mode = st.selectbox("Select Change-Point Model Type:",["auto", "heating", "cooling"],
                        index=0, disabled=mode_disabled)
                else:
                    mode = st.selectbox("Select Change-Point Model Type:",["auto", "heating", "cooling"],index=0)


            if st.button("Calculate Savings"):
                three_res = None
                five_res = None

                if model_choice == "3-parameter":
                    three_res = fit_three_param_cp(temp, energy, Tmin, Tmax, step, mode = mode)

                if model_choice == "5-parameter":
                    five_res = fit_five_param_deadband(temp, energy, Tmin, Tmax, step)

                if model_choice == "Both":
                    three_res = fit_three_param_cp(temp, energy, Tmin, Tmax, step, mode = mode)
                    five_res = fit_five_param_deadband(temp, energy, Tmin, Tmax, step)


                mean_energy = float(final_df['Energy'].mean())
                #preferred_label, preferred_result = select_model_by_rmse_r2(three_res, five_res, rel_tol_pct, mean_kwh)


                # -------------------------
                # DISPLAY RESULTS
                # -------------------------
                st.write("## Model Results")

                if model_choice in ["3-parameter"]:
                    three_para_results(three_res,temperature_unit,mean_energy)

                if model_choice in ["5-parameter"]:
                    five_para_results(five_res,temperature_unit,mean_energy)

                if model_choice in ["Both"]:
                    three_five_para_results(three_res,five_res,temperature_unit,mean_energy)


                # -------------------------
                # PLOT MODELS
                # -------------------------
                T_plot = np.linspace(final_df['Temperature'].min(), final_df['Temperature'].max(), 400)


                fig, ax = plt.subplots(figsize=(9, 5))
                ax.scatter(final_df['Temperature'], final_df['Energy'], label="Measured Energy", s=50)

                if model_choice == "3-parameter":
                    Y3_plot = predict_3p_for_plot(T_plot, three_res["Tb"], three_res["model"], mode = three_res["mode"])
                    ax.plot(T_plot, Y3_plot, label="3-parameter", linewidth=2.5)

                elif model_choice == "5-parameter":
                    Y5_plot = predict_5p_for_plot(T_plot, five_res["Tb_low"], five_res["Tb_high"], five_res["model"])
                    ax.plot(T_plot, Y5_plot, label="5-parameter", linewidth=2.5)

                else:  # Both
                    Y3_plot = predict_3p_for_plot(T_plot, three_res["Tb"], three_res["model"], mode = three_res["mode"])
                    Y5_plot = predict_5p_for_plot(T_plot, five_res["Tb_low"], five_res["Tb_high"], five_res["model"])
                    ax.plot(T_plot, Y3_plot, label="3-parameter", linewidth=2.5)
                    ax.plot(T_plot, Y5_plot, label="5-parameter", linewidth=2.5)

                # Deadband shade
                if model_choice in ["5-parameter", "Both"]:
                    ax.axvspan(five_res["Tb_low"], five_res["Tb_high"], alpha=0.08, color="gray", label="Deadband")

                if temperature_unit == "celsius":
                    ax.set_xlabel("Temperature (°C)")
                else:
                    ax.set_xlabel('Temperature (°F)')
                ax.set_ylabel("Energy")
                ax.set_title("3-Parameter vs 5-Parameter Change-Point Models")
                ax.legend()
                ax.grid(True)

                st.pyplot(fig)

                # --------------------------

                y_r = reported_df['Energy']
                x_r = reported_df['Temperature']

                if model_choice == "3-parameter":
                    pred_r = predict_3p_for_plot(x_r.to_numpy(), three_res["Tb"], three_res["model"],
                                                 mode=three_res["mode"])

                elif model_choice == "5-parameter":
                    pred_r = predict_5p_for_plot(x_r.to_numpy(), five_res["Tb_low"], five_res["Tb_high"],
                                                 five_res["model"])

                else:  # Both
                    pred_r_3p = predict_3p_for_plot(x_r.to_numpy(), three_res["Tb"], three_res["model"],
                                                    mode=three_res["mode"])
                    pred_r_5p = predict_5p_for_plot(x_r.to_numpy(), five_res["Tb_low"], five_res["Tb_high"],
                                                    five_res["model"])

                if model_choice == "3-parameter" or model_choice == "5-parameter":
                    st.write(f'##### Predicted Baseline Consumption: \n {pred_r.sum():.2f}')

                else:
                    mod1, mod2 = st.columns(2, border=True)
                    with mod1:
                        st.write(f'##### 3 Parameter Predicted Baseline Consumption: \n {pred_r_3p.sum():.2f}')
                    with mod2:
                        st.write(f'##### 5 Parameter Predicted Baseline Consumption: \n {pred_r_5p.sum():.2f}')

                st.write(f'##### Reported Consumption: \n {y_r.sum():.2f}')

                if model_choice == "3-parameter" or model_choice == "5-parameter":
                    savings = pred_r.sum() - y_r.sum()
                    st.write(f'##### Savings: \n {savings:.2f}')
                else:
                    mod1, mod2 = st.columns(2, border=True)

                    with mod1:
                        savings_3p = pred_r_3p.sum() - y_r.sum()
                        st.write(f'##### 3 Parameter Savings: \n {savings_3p:.2f}')
                    with mod2:
                        savings_5p = pred_r_5p.sum() - y_r.sum()
                        st.write(f'##### 5 Parameter Savings: \n {savings_5p:.2f}')

    elif manual_data == 'Yes':

        interval = st.radio('Do you want Weather Data using Specific Intervals?', options=["Yes", "No"], index=None, key='interval_data')

        if interval == 'Yes':

            col1, col2 = st.columns(2, border=True)

            with col1:
                daily_hourly = st.radio('Do you have Daily or Hourly Intervals?', options=["Daily", "Hourly"], index=None, key='daily_hourly')

            with col2:
                tmy = st.radio('Do you want to Normalize Savings using TMY?', options=["Yes", "No"], index=None, key='tmy_data')

            if tmy == 'Yes':

                st.write('#### Enter Location Details')

                lat, lon, temp = st.columns(3, border=True)

                with lat:
                    lat = st.number_input("Latitude", format="%.4f")

                with lon:
                    lon = st.number_input("Longitude", format="%.4f")

                with temp:
                    temperature_unit = st.selectbox('Select Temperature Unit:',['celsius','fahrenheit'])

                var = "temperature"  # or let user pick
                which = "hourly"

                # create client once (you can cache it)

                try:
                    weather_tmy = pvgis_tmy(lat, lon)
                    weather_tmy['month'] = weather_tmy['time'].dt.month
                    avg_weather_tmy = weather_tmy.groupby('month')['temperature'].mean().reset_index()
                    if temperature_unit == "celsius":
                        st.write(avg_weather_tmy)
                    else:
                        avg_weather_tmy['temperature'] = (avg_weather_tmy['temperature'] * 9/5) + 32
                        st.write(avg_weather_tmy)
                except requests.exceptions.HTTPError:
                    st.error(f'Please Enter Correct Latitude and Longitude')


                client = make_openmeteo_client()

                st.write('#### Select Model Details')

                model_choice, mode, step = model_details()

                timezone = get_timezone_from_coords(lat, lon)

                # Build column names automatically
                empty_df = pd.DataFrame({'Start Date (yyyy-mm-dd)': pd.to_datetime(['2025-01-01']),
                                         'End Date (yyyy-mm-dd)': pd.to_datetime(['2025-02-01']),
                                         'Energy': pd.Series([0.0], dtype='float64')})

                st.write('#### Enter Baseline Energy Data Below:')
                baseline_df = st.data_editor(empty_df, num_rows="dynamic", key='baseline_energy')

                if len(baseline_df) < 2:
                    st.error("Please enter at least 2 rows.")

                st.write('#### Enter Reported Energy Data Below:')
                reported_df = st.data_editor(empty_df, num_rows="dynamic", key='reported_energy')

                date_cols = ['Start Date (yyyy-mm-dd)', 'End Date (yyyy-mm-dd)']

                for col in date_cols:
                    baseline_df[col] = (pd.to_datetime(baseline_df[col]).dt.tz_localize(timezone))
                    reported_df[col] = (pd.to_datetime(reported_df[col]).dt.tz_localize(timezone))

                if len(baseline_df) >= 2:

                    if st.button("Fetch Weather Data & Calculate Savings"):

                        with st.spinner('Calculating...'):

                            for i in range(len(baseline_df)):
                                start_dt = baseline_df.loc[i, 'Start Date (yyyy-mm-dd)']
                                end_dt = baseline_df.loc[i, 'End Date (yyyy-mm-dd)']

                                start_date = start_dt.date().isoformat()
                                end_date = end_dt.date().isoformat()
                                meta, temperature_data = fetch_openmeteo_archive(client, lat, lon, start_date, end_date, temperature_unit, which, var)

                                temperature_data['date_local'] = pd.to_datetime(temperature_data['date_local'])

                                mask = ((temperature_data['date_local'] >= start_dt)&(temperature_data['date_local'] <= end_dt))

                                filtered_temp = temperature_data.loc[mask]

                                if daily_hourly == 'Daily':
                                    filtered_temp['date'] = filtered_temp['date_local'].dt.date

                                    hourly_avg = (filtered_temp.groupby(['date'], as_index=False)['temperature'].mean())

                                else:
                                    filtered_temp['date'] = filtered_temp['date_local'].dt.date
                                    filtered_temp['hour'] = filtered_temp['date_local'].dt.hour

                                    hourly_avg = (filtered_temp.groupby(['date', 'hour'], as_index=False)['temperature'].mean())

                                baseline_df.loc[i, 'temperature'] = hourly_avg["temperature"].mean()
                                baseline_df['days'] = end_dt - start_dt
                            st.write(baseline_df)

                            for i in range(len(reported_df)):
                                start_dt = reported_df.loc[i, 'Start Date (yyyy-mm-dd)']
                                end_dt = reported_df.loc[i, 'End Date (yyyy-mm-dd)']

                                start_date = start_dt.date().isoformat()
                                end_date = end_dt.date().isoformat()
                                meta, temperature_data = fetch_openmeteo_archive(client, lat, lon, start_date, end_date, temperature_unit, which, var)

                                temperature_data['date_local'] = pd.to_datetime(temperature_data['date_local'])

                                mask = ((temperature_data['date_local'] >= start_dt) & (temperature_data['date_local'] <= end_dt))

                                filtered_temp = temperature_data.loc[mask]

                                if daily_hourly == 'Daily':
                                    filtered_temp['date'] = filtered_temp['date_local'].dt.date

                                    hourly_avg = (filtered_temp.groupby(['date'], as_index=False)['temperature'].mean())

                                else:
                                    filtered_temp['date'] = filtered_temp['date_local'].dt.date
                                    filtered_temp['hour'] = filtered_temp['date_local'].dt.hour

                                    hourly_avg = (filtered_temp.groupby(['date', 'hour'], as_index=False)['temperature'].mean())

                                reported_df.loc[i, 'temperature'] = hourly_avg["temperature"].mean()


                            # -------------------------
                            # RUN MODELS FOR BASELINE
                            # -------------------------

                            Tmin_b = float(np.floor(baseline_df['temperature'].min()))
                            Tmax_b = float(np.ceil(baseline_df['temperature'].max()))

                            temp_b = baseline_df['temperature'].values
                            energy_b = baseline_df['Energy'].values

                            three_res_b = None
                            five_res_b = None

                            if model_choice == "3-parameter":
                                three_res_b = fit_three_param_cp(temp_b, energy_b, Tmin_b, Tmax_b, step, mode=mode)

                            if model_choice == "5-parameter":
                                five_res_b = fit_five_param_deadband(temp_b, energy_b, Tmin_b, Tmax_b, step)

                            if model_choice == "Both":
                                three_res_b = fit_three_param_cp(temp_b, energy_b, Tmin_b, Tmax_b, step, mode=mode)
                                five_res_b = fit_five_param_deadband(temp_b, energy_b, Tmin_b, Tmax_b, step)

                            mean_energy_b = float(baseline_df['Energy'].mean())
                            # preferred_label, preferred_result = select_model_by_rmse_r2(three_res, five_res, rel_tol_pct,mean_energy)

                            # -------------------------
                            # RUN MODELS FOR REPORTED
                            # -------------------------

                            Tmin_r = float(np.floor(reported_df['temperature'].min()))
                            Tmax_r = float(np.ceil(reported_df['temperature'].max()))

                            temp_r = reported_df['temperature'].values
                            energy_r = reported_df['Energy'].values

                            three_res_r = None
                            five_res_r = None

                            if model_choice == "3-parameter":
                                three_res_r = fit_three_param_cp(temp_r, energy_r, Tmin_r, Tmax_r, step, mode=mode)

                            if model_choice == "5-parameter":
                                five_res_r = fit_five_param_deadband(temp_r, energy_r, Tmin_r, Tmax_r, step)

                            if model_choice == "Both":
                                three_res_r = fit_three_param_cp(temp_r, energy_r, Tmin_r, Tmax_r, step, mode=mode)
                                five_res_r = fit_five_param_deadband(temp_r, energy_r, Tmin_r, Tmax_r, step)

                            mean_energy_r = float(reported_df['Energy'].mean())
                            # preferred_label, preferred_result = select_model_by_rmse_r2(three_res, five_res, rel_tol_pct,mean_energy)


                            # ----------------------------
                            # DISPLAY RESULTS FOR BASELINE
                            # ----------------------------
                            st.write("## Baseline Model Results")

                            if model_choice in ["3-parameter"]:
                                three_para_results(three_res_b, temperature_unit, mean_energy_b)

                            if model_choice in ["5-parameter"]:
                                five_para_results(five_res_b, temperature_unit, mean_energy_b)

                            if model_choice in ["Both"]:
                                three_five_para_results(three_res_b, five_res_b, temperature_unit, mean_energy_b)


                            # -------------------------
                            # PLOT MODELS FOR BASELINE
                            # -------------------------
                            T_plot = np.linspace(baseline_df['temperature'].min(), baseline_df['temperature'].max(), 400)

                            fig, ax = plt.subplots(figsize=(9, 5))
                            ax.scatter(baseline_df['temperature'], baseline_df['Energy'], label="Measured Energy", s=50)

                            if model_choice == "3-parameter":
                                Y3_plot = predict_3p_for_plot(T_plot, three_res_b["Tb"], three_res_b["model"],
                                                              mode=three_res_b["mode"])
                                ax.plot(T_plot, Y3_plot, label="3-parameter", linewidth=2.5)

                            elif model_choice == "5-parameter":
                                Y5_plot = predict_5p_for_plot(T_plot, five_res_b["Tb_low"], five_res_b["Tb_high"],
                                                              five_res_b["model"])
                                ax.plot(T_plot, Y5_plot, label="5-parameter", linewidth=2.5)

                            else:  # Both
                                Y3_plot = predict_3p_for_plot(T_plot, three_res_b["Tb"], three_res_b["model"],
                                                              mode=three_res_b["mode"])
                                Y5_plot = predict_5p_for_plot(T_plot, five_res_b["Tb_low"], five_res_b["Tb_high"],
                                                              five_res_b["model"])
                                ax.plot(T_plot, Y3_plot, label="3-parameter", linewidth=2.5)
                                ax.plot(T_plot, Y5_plot, label="5-parameter", linewidth=2.5)

                            # Deadband shade
                            if model_choice in ["5-parameter", "Both"]:
                                ax.axvspan(five_res_b["Tb_low"], five_res_b["Tb_high"], alpha=0.08, color="gray",
                                           label="Deadband")

                            if temperature_unit == "celsius":
                                ax.set_xlabel("Temperature (°C)")
                            else:
                                ax.set_xlabel("Temperature (°F)")
                            ax.set_ylabel("Energy")

                            if model_choice in ["3-parameter"]:
                                ax.set_title("3-Parameter Change-Point Model")
                            elif model_choice in ["5-parameter"]:
                                ax.set_title("5-Parameter Change-Point Model")
                            else:
                                ax.set_title("3-Parameter and 5-parameter Change-Point Models")

                            ax.legend()
                            ax.grid(True)

                            st.pyplot(fig)

                            # ----------------------------
                            # DISPLAY RESULTS FOR REPORTED
                            # ----------------------------
                            st.write("## Reported Model Results")

                            if model_choice in ["3-parameter"]:
                                three_para_results(three_res_r, temperature_unit, mean_energy_r)

                            if model_choice in ["5-parameter"]:
                                five_para_results(five_res_r, temperature_unit, mean_energy_r)

                            if model_choice in ["Both"]:
                                three_five_para_results(three_res_r, five_res_r, temperature_unit, mean_energy_r)


                            # -------------------------
                            # PLOT MODELS FOR REPORTED
                            # -------------------------
                            T_plot = np.linspace(reported_df['temperature'].min(), reported_df['temperature'].max(),400)

                            fig, ax = plt.subplots(figsize=(9, 5))
                            ax.scatter(reported_df['temperature'], reported_df['Energy'], label="Measured Energy", s=50)

                            if model_choice == "3-parameter":
                                Y3_plot = predict_3p_for_plot(T_plot, three_res_r["Tb"], three_res_r["model"],
                                                              mode=three_res_r["mode"])
                                ax.plot(T_plot, Y3_plot, label="3-parameter", linewidth=2.5)

                            elif model_choice == "5-parameter":
                                Y5_plot = predict_5p_for_plot(T_plot, five_res_r["Tb_low"], five_res_r["Tb_high"],
                                                              five_res_r["model"])
                                ax.plot(T_plot, Y5_plot, label="5-parameter", linewidth=2.5)

                            else:  # Both
                                Y3_plot = predict_3p_for_plot(T_plot, three_res_r["Tb"], three_res_r["model"],
                                                              mode=three_res_r["mode"])
                                Y5_plot = predict_5p_for_plot(T_plot, five_res_b["Tb_low"], five_res_r["Tb_high"],
                                                              five_res_r["model"])
                                ax.plot(T_plot, Y3_plot, label="3-parameter", linewidth=2.5)
                                ax.plot(T_plot, Y5_plot, label="5-parameter", linewidth=2.5)

                            # Deadband shade
                            if model_choice in ["5-parameter", "Both"]:
                                ax.axvspan(five_res_r["Tb_low"], five_res_r["Tb_high"], alpha=0.08, color="gray",
                                           label="Deadband")

                            if temperature_unit == "celsius":
                                ax.set_xlabel("Temperature (°C)")
                            else:
                                ax.set_xlabel("Temperature (°F)")
                            ax.set_ylabel("Energy")

                            if model_choice in ["3-parameter"]:
                                ax.set_title("3-Parameter Change-Point Model")
                            elif model_choice in ["5-parameter"]:
                                ax.set_title("5-Parameter Change-Point Model")
                            else:
                                ax.set_title("3-Parameter and 5-parameter Change-Point Models")

                            ax.legend()
                            ax.grid(True)

                            st.pyplot(fig)


                            # TMY Calculations--------------------------

                            x_tmy = avg_weather_tmy['temperature']

                            if model_choice == "3-parameter":
                                pred_base_tmy = predict_3p_for_plot(x_tmy.to_numpy(), three_res_b["Tb"], three_res_b["model"],
                                                             mode=three_res_b["mode"])
                                pred_rep_tmy = predict_3p_for_plot(x_tmy.to_numpy(), three_res_r["Tb"],
                                                                    three_res_r["model"],
                                                                    mode=three_res_r["mode"])

                            elif model_choice == "5-parameter":
                                pred_base_tmy = predict_5p_for_plot(x_tmy.to_numpy(), five_res_b["Tb_low"], five_res_b["Tb_high"],
                                                             five_res_b["model"])
                                pred_rep_tmy = predict_5p_for_plot(x_tmy.to_numpy(), five_res_r["Tb_low"],
                                                                    five_res_r["Tb_high"],
                                                                    five_res_r["model"])

                            else:  # Both
                                pred_base_tmy_3p = predict_3p_for_plot(x_tmy.to_numpy(), three_res_b["Tb"], three_res_b["model"],
                                                                mode=three_res_b["mode"])
                                pred_rep_tmy_3p = predict_3p_for_plot(x_tmy.to_numpy(), three_res_r["Tb"],
                                                                       three_res_r["model"],
                                                                       mode=three_res_r["mode"])
                                pred_base_tmy_5p = predict_5p_for_plot(x_tmy.to_numpy(), five_res_b["Tb_low"], five_res_b["Tb_high"],
                                                                five_res_b["model"])
                                pred_rep_tmy_5p = predict_5p_for_plot(x_tmy.to_numpy(), five_res_r["Tb_low"],
                                                                       five_res_r["Tb_high"],
                                                                       five_res_r["model"])

                            if model_choice == "3-parameter" or model_choice == "5-parameter":
                                st.write(f'##### Predicted Baseline Consumption: \n {pred_base_tmy.sum():.2f}')
                                st.write(f'##### Predicted Reported Consumption: \n {pred_rep_tmy.sum():.2f}')

                            else:
                                mod1, mod2 = st.columns(2, border=True)
                                with mod1:
                                    st.write(f'##### 3 Parameter Predicted Baseline Consumption: \n {pred_base_tmy_3p.sum():.2f}')
                                    st.write(f'##### 3 Parameter Predicted Reported Consumption: \n {pred_rep_tmy_3p.sum():.2f}')
                                with mod2:
                                    st.write(f'##### 5 Parameter Predicted Baseline Consumption: \n {pred_base_tmy_5p.sum():.2f}')
                                    st.write(f'##### 5 Parameter Predicted Reported Consumption: \n {pred_rep_tmy_5p.sum():.2f}')


                            if model_choice == "3-parameter" or model_choice == "5-parameter":
                                savings = pred_base_tmy.sum() - pred_rep_tmy.sum()
                                st.write(f'##### Savings: \n {savings:.2f}')
                            else:
                                mod1, mod2 = st.columns(2, border=True)

                                with mod1:
                                    savings_3p = pred_base_tmy_3p.sum() - pred_rep_tmy_3p.sum()
                                    st.write(f'##### 3 Parameter Savings: \n {savings_3p:.2f}')
                                with mod2:
                                    savings_5p = pred_base_tmy_5p.sum() - pred_rep_tmy_5p.sum()
                                    st.write(f'##### 5 Parameter Savings: \n {savings_5p:.2f}')

            if tmy == 'No':

                st.write('#### Enter Location Details')

                lat, lon, temp = st.columns(3, border=True)

                with lat:
                    lat = st.number_input("Latitude", format="%.4f")

                with lon:
                    lon = st.number_input("Longitude", format="%.4f")

                with temp:
                    temperature_unit = st.selectbox('Select Temperature Unit:', ['celsius', 'fahrenheit'])

                var = "temperature"  # or let user pick
                which = "hourly"

                # create client once (you can cache it)

                client = make_openmeteo_client()

                st.write('#### Select Model Details')

                model_choice, mode, step = model_details()

                timezone = get_timezone_from_coords(lat, lon)

                # Build column names automatically
                empty_df = pd.DataFrame(
                    {'Start Date (yyyy-mm-dd)': pd.to_datetime(['2025-01-01']),
                     'End Date (yyyy-mm-dd)': pd.to_datetime(['2025-02-01']),
                     'Energy': pd.Series([0.0], dtype='float64')})

                st.write('#### Enter Baseline Energy Data Below:')
                baseline_df = st.data_editor(empty_df, num_rows="dynamic", key='baseline_energy')


                if len(baseline_df) < 2:
                    st.error("Please enter at least 2 rows.")

                st.write('#### Enter Reported Energy Data Below:')
                reported_df = st.data_editor(empty_df, num_rows="dynamic", key='reported_energy')

                date_cols = ['Start Date (yyyy-mm-dd)', 'End Date (yyyy-mm-dd)']

                for col in date_cols:
                    baseline_df[col] = (pd.to_datetime(baseline_df[col]).dt.tz_localize(timezone))
                    reported_df[col] = (pd.to_datetime(reported_df[col]).dt.tz_localize(timezone))


                if len(baseline_df) >= 2:

                    if st.button("Fetch Weather Data & Calculate Savings"):

                        with st.spinner('Calculating...'):

                            for i in range(len(baseline_df)):
                                start_dt = baseline_df.loc[i, 'Start Date (yyyy-mm-dd)']
                                end_dt = baseline_df.loc[i, 'End Date (yyyy-mm-dd)']

                                start_date = start_dt.date().isoformat()
                                end_date = end_dt.date().isoformat()
                                meta, temperature_data = fetch_openmeteo_archive(client, lat, lon, start_date, end_date,
                                                                                 temperature_unit, which, var)

                                temperature_data['date_local'] = pd.to_datetime(temperature_data['date_local'])

                                mask = ((temperature_data['date_local'] >= start_dt) & (temperature_data['date_local'] <= end_dt))

                                filtered_temp = temperature_data.loc[mask]

                                if daily_hourly == 'Daily':
                                    filtered_temp['date'] = filtered_temp['date_local'].dt.date

                                    hourly_avg = (filtered_temp.groupby(['date'], as_index=False)['temperature'].mean())

                                else:
                                    filtered_temp['date'] = filtered_temp['date_local'].dt.date
                                    filtered_temp['hour'] = filtered_temp['date_local'].dt.hour

                                    hourly_avg = (filtered_temp.groupby(['date', 'hour'], as_index=False)['temperature'].mean())

                                baseline_df.loc[i, 'temperature'] = hourly_avg["temperature"].mean()

                            for i in range(len(reported_df)):
                                start_dt = reported_df.loc[i, 'Start Date (yyyy-mm-dd)'].tz_convert(timezone)
                                end_dt = reported_df.loc[i, 'End Date (yyyy-mm-dd)'].tz_convert(timezone)

                                start_date = start_dt.date().isoformat()
                                end_date = end_dt.date().isoformat()
                                meta, temperature_data = fetch_openmeteo_archive(client, lat, lon, start_date, end_date,
                                                                                 temperature_unit, which, var)

                                temperature_data['date_local'] = pd.to_datetime(temperature_data['date_local'])

                                mask = ((temperature_data['date_local'] >= start_dt) & (temperature_data['date_local'] <= end_dt))

                                filtered_temp = temperature_data.loc[mask]

                                if daily_hourly == 'Daily':
                                    filtered_temp['date'] = filtered_temp['date_local'].dt.date

                                    hourly_avg = (filtered_temp.groupby(['date'], as_index=False)['temperature'].mean())

                                else:
                                    filtered_temp['date'] = filtered_temp['date_local'].dt.date
                                    filtered_temp['hour'] = filtered_temp['date_local'].dt.hour

                                    hourly_avg = (
                                        filtered_temp.groupby(['date', 'hour'], as_index=False)['temperature'].mean())

                                reported_df.loc[i, 'temperature'] = hourly_avg["temperature"].mean()


                            # -------------------------
                            # DEFAULT MODEL SETTINGS
                            # -------------------------

                            Tmin = float(np.floor(baseline_df['temperature'].min()))
                            Tmax = float(np.ceil(baseline_df['temperature'].max()))

                            # -------------------------
                            # RUN MODELS
                            # -------------------------
                            temp = baseline_df['temperature'].values
                            energy = baseline_df['Energy'].values

                            three_res = None
                            five_res = None

                            if model_choice == "3-parameter":
                                three_res = fit_three_param_cp(temp, energy, Tmin, Tmax, step, mode=mode)

                            if model_choice == "5-parameter":
                                five_res = fit_five_param_deadband(temp, energy, Tmin, Tmax, step)

                            if model_choice == "Both":
                                three_res = fit_three_param_cp(temp, energy, Tmin, Tmax, step, mode=mode)
                                five_res = fit_five_param_deadband(temp, energy, Tmin, Tmax, step)

                            mean_energy = float(baseline_df['Energy'].mean())
                            # preferred_label, preferred_result = select_model_by_rmse_r2(three_res, five_res, rel_tol_pct,mean_energy)

                            # -------------------------
                            # DISPLAY RESULTS
                            # -------------------------
                            st.write("## Model Results")

                            if model_choice in ["3-parameter"]:
                                three_para_results(three_res, temperature_unit, mean_energy)

                            if model_choice in ["5-parameter"]:
                                five_para_results(five_res, temperature_unit, mean_energy)

                            if model_choice in ["Both"]:
                                three_five_para_results(three_res, five_res, temperature_unit, mean_energy)

                            # -------------------------
                            # PLOT MODELS
                            # -------------------------
                            T_plot = np.linspace(baseline_df['temperature'].min(), baseline_df['temperature'].max(), 400)

                            fig, ax = plt.subplots(figsize=(9, 5))
                            ax.scatter(baseline_df['temperature'], baseline_df['Energy'], label="Measured Energy", s=50)

                            if model_choice == "3-parameter":
                                Y3_plot = predict_3p_for_plot(T_plot, three_res["Tb"], three_res["model"],
                                                              mode=three_res["mode"])
                                ax.plot(T_plot, Y3_plot, label="3-parameter", linewidth=2.5)

                            elif model_choice == "5-parameter":
                                Y5_plot = predict_5p_for_plot(T_plot, five_res["Tb_low"], five_res["Tb_high"],
                                                              five_res["model"])
                                ax.plot(T_plot, Y5_plot, label="5-parameter", linewidth=2.5)

                            else:  # Both
                                Y3_plot = predict_3p_for_plot(T_plot, three_res["Tb"], three_res["model"],
                                                              mode=three_res["mode"])
                                Y5_plot = predict_5p_for_plot(T_plot, five_res["Tb_low"], five_res["Tb_high"],
                                                              five_res["model"])
                                ax.plot(T_plot, Y3_plot, label="3-parameter", linewidth=2.5)
                                ax.plot(T_plot, Y5_plot, label="5-parameter", linewidth=2.5)

                            # Deadband shade
                            if model_choice in ["5-parameter", "Both"]:
                                ax.axvspan(five_res["Tb_low"], five_res["Tb_high"], alpha=0.08, color="gray",
                                           label="Deadband")

                            if temperature_unit == "celsius":
                                ax.set_xlabel("Temperature (°C)")
                            else:
                                ax.set_xlabel("Temperature (°F)")
                            ax.set_ylabel("Energy")
                            ax.set_title("3-Parameter vs 5-Parameter Change-Point Models")
                            ax.legend()
                            ax.grid(True)

                            st.pyplot(fig)

                            # --------------------------

                            y_r = reported_df['Energy']
                            x_r = reported_df['temperature']

                            if model_choice == "3-parameter":
                                pred_r = predict_3p_for_plot(x_r.to_numpy(), three_res["Tb"], three_res["model"],
                                                             mode=three_res["mode"])

                            elif model_choice == "5-parameter":
                                pred_r = predict_5p_for_plot(x_r.to_numpy(), five_res["Tb_low"], five_res["Tb_high"],
                                                             five_res["model"])

                            else:  # Both
                                pred_r_3p = predict_3p_for_plot(x_r.to_numpy(), three_res["Tb"], three_res["model"],
                                                                mode=three_res["mode"])
                                pred_r_5p = predict_5p_for_plot(x_r.to_numpy(), five_res["Tb_low"], five_res["Tb_high"],
                                                                five_res["model"])

                            if model_choice == "3-parameter" or model_choice == "5-parameter":
                                st.write(f'##### Predicted Baseline Consumption: \n {pred_r.sum():.2f}')

                            else:
                                mod1, mod2 = st.columns(2, border=True)
                                with mod1:
                                    st.write(
                                        f'##### 3 Parameter Predicted Baseline Consumption: \n {pred_r_3p.sum():.2f}')
                                with mod2:
                                    st.write(
                                        f'##### 5 Parameter Predicted Baseline Consumption: \n {pred_r_5p.sum():.2f}')

                            st.write(f'##### Reported Consumption: \n {y_r.sum():.2f}')

                            if model_choice == "3-parameter" or model_choice == "5-parameter":
                                savings = pred_r.sum() - y_r.sum()
                                st.write(f'##### Savings: \n {savings:.2f}')
                            else:
                                mod1, mod2 = st.columns(2, border=True)

                                with mod1:
                                    savings_3p = pred_r_3p.sum() - y_r.sum()
                                    st.write(f'##### 3 Parameter Savings: \n {savings_3p:.2f}')
                                with mod2:
                                    savings_5p = pred_r_5p.sum() - y_r.sum()
                                    st.write(f'##### 5 Parameter Savings: \n {savings_5p:.2f}')

        if interval == 'No':

            tmy = st.radio('Do you want to Normalize Savings using TMY?', options=["Yes", "No"], index=None,
                           key='tmy_data')

            if tmy == "Yes" or tmy == "No":

                # Build column names automatically
                empty_df = pd.DataFrame({"Energy": pd.Series([0], dtype=float)})

                base, reported = st.columns(2)

                with base:
                    st.write('##### Enter Baseline Energy Data Below:')

                    manual_df = st.data_editor(empty_df, num_rows="dynamic", key='baseline_dataframe')

                    if len(manual_df) < 2:
                        st.error("Please enter at least 2 rows.")

                with reported:
                    st.write('##### Enter Reported Energy Data Below:')

                    reported_df = st.data_editor(empty_df, num_rows="dynamic", key='reported_dataframe')



                # Fetch Weather Data

                st.write('#### Select Baseline Period Dates')

                start_b, end_b, weather_int_b = st.columns(3, border=True)

                with start_b:
                    start_date_b = st.date_input("Start date", value=date.today() - timedelta(days=365*2+1))
                with end_b:
                    end_date_b = st.date_input("End date", value=date.today() - timedelta(days=366))
                with weather_int_b:
                    weather_interval = st.selectbox('Select Interval', {'Hourly', 'Daily', 'Monthly'})

                st.write('#### Select Reported Period Dates')

                start_r, end_r, weather_int_r = st.columns(3, border=True)

                with start_r:
                    start_date_r = st.date_input("Start date", value=date.today() - timedelta(days=365))
                with end_r:
                    end_date_r = st.date_input("End date", value=date.today() - timedelta(days=1))
                with weather_int_r:
                    st.selectbox('Select Interval', weather_interval, disabled=True, help='This will be same as baseline interval to avoid errors.')


                st.write('#### Enter Location Details')

                lat, lon, temp = st.columns(3, border=True)

                with lat:
                    lat = st.number_input("Latitude", format="%.4f")


                with lon:
                    lon = st.number_input("Longitude", format="%.4f")


                with temp:
                    temperature_unit = st.selectbox('Select Temperature Unit:', ['celsius', 'fahrenheit'])


                var = "temperature"  # or let user pick
                which = "hourly"


                # create client once (you can cache it)

                client = make_openmeteo_client()

                st.write('#### Select Model Details')

                weather_i, model_c, step_unit = st.columns(3, border=True)

                with weather_i:
                    model_choice = st.selectbox("Select Change-Point Model:", ["3-parameter", "5-parameter", "Both"])

                with model_c:

                    if model_choice == "3-parameter":
                        mode = st.selectbox("Select Change-Point Model Type:", ["auto", "heating", "cooling"],
                                            index=0)
                    elif model_choice == "5-parameter":
                        # Disable the mode selection if the model is not "3-parameter"
                        mode_disabled = model_choice != "3-parameter"

                        mode = st.selectbox("Select Change-Point Model Type:", ["auto", "heating", "cooling"],
                                            index=0, disabled=mode_disabled)
                    else:
                        mode = st.selectbox("Select Change-Point Model Type:", ["auto", "heating", "cooling"],
                                            index=0)

                with step_unit:
                    step = st.selectbox('Select Decimals for Balance Point:', [1.0,0.5,0.1], help='This interval will be used to calculate Balance Point.'
                                                                                                 '\n E.g., If you select 1.0, balance point will be calculated between 55, 56, 57, and so on.'
                                                                                                  '\n If you select 0.5, it will be calculated between 55.0, 55.5, 56.0, and so on.')

                if tmy == 'Yes':

                    try:
                        weather_tmy = pvgis_tmy(lat, lon)
                        weather_tmy['month'] = weather_tmy['time'].dt.month
                        avg_weather_tmy = weather_tmy.groupby('month')['temperature'].mean().reset_index()
                        if temperature_unit == "celsius":
                            st.write(avg_weather_tmy)
                        else:
                            avg_weather_tmy['temperature'] = (avg_weather_tmy['temperature'] * 9 / 5) + 32
                            st.write(avg_weather_tmy)
                    except requests.exceptions.HTTPError:
                        st.error(f'Please Enter Correct Latitude and Longitude')

                    if st.button("Fetch Weather Data & Calculate savings"):
                        if start_date_b > end_date_b and start_date_r > end_date_r:
                            st.error("Start must be <= end")
                        else:
                            start_str_b = start_date_b.isoformat()
                            end_str_b = end_date_b.isoformat()
                            start_str_r = start_date_r.isoformat()
                            end_str_r = end_date_r.isoformat()
                            with st.spinner("Calculating..."):
                                try:
                                    meta, df_weather = fetch_openmeteo_archive(client, lat, lon, start_str_b, end_str_b, temperature_unit, which, var)
                                    meta, df_weather_r = fetch_openmeteo_archive(client, lat, lon, start_str_r, end_str_r, temperature_unit, which, var)
                                except Exception as e:
                                    st.error(f"Weather fetch failed: {e}")
                                else:
                                    st.success("Fetched weather data")

                                    df_temp = df_weather.copy()
                                    df_temp_r = df_weather_r.copy()

                                    if weather_interval == "Hourly":

                                        df_weather_final = df_temp
                                        df_weather_final_r = df_temp_r

                                    elif weather_interval == "Monthly":

                                        df_temp['month'] = df_temp['date_local'].dt.month  # 1–12
                                        df_temp_r['month'] = df_temp_r['date_local'].dt.month

                                        df_weather_final = (df_temp.groupby('month', as_index=False).mean(numeric_only=True))
                                        df_weather_final_r = (df_temp_r.groupby('month', as_index=False).mean(numeric_only=True))

                                    else:

                                        df_temp['month'] = df_temp['date_local'].dt.month
                                        df_temp['day'] = df_temp['date_local'].dt.day
                                        df_temp_r['month'] = df_temp_r['date_local'].dt.month
                                        df_temp_r['day'] = df_temp_r['date_local'].dt.day

                                        df_weather_final = (
                                            df_temp.groupby(['month', 'day'], as_index=False).mean(numeric_only=True))
                                        df_weather_final_r = (
                                            df_temp_r.groupby(['month', 'day'], as_index=False).mean(numeric_only=True))


                                baseline_df = pd.concat([manual_df, df_weather_final], axis=1)
                                reported_df = pd.concat([reported_df, df_weather_final_r], axis=1)

                                # -------------------------
                                # DEFAULT MODEL SETTINGS
                                # -------------------------

                                Tmin_b = float(np.floor(baseline_df['temperature'].min()))
                                Tmax_b = float(np.ceil(baseline_df['temperature'].max()))

                                Tmin_r = float(np.floor(reported_df['temperature'].min()))
                                Tmax_r = float(np.ceil(reported_df['temperature'].max()))

                                # -------------------------
                                # RUN MODELS FOR BASELINE
                                # -------------------------
                                temp_b = baseline_df['temperature'].dropna().values
                                energy_b = baseline_df['Energy'].dropna().values

                                if len(temp_b) == len(energy_b):

                                    with st.spinner('Running change-point models...'):
                                        three_res_b = None
                                        five_res_b = None

                                        if model_choice == '3-parameter':
                                            three_res_b = fit_three_param_cp(temp_b, energy_b, Tmin_b, Tmax_b, step, mode=mode)

                                        if model_choice == '5-parameter':
                                            five_res_b = fit_five_param_deadband(temp_b, energy_b, Tmin_b, Tmax_b, step)

                                        if model_choice == 'Both':
                                            three_res_b = fit_three_param_cp(temp_b, energy_b, Tmin_b, Tmax_b, step, mode=mode)
                                            five_res_b = fit_five_param_deadband(temp_b, energy_b, Tmin_b, Tmax_b, step)

                                    mean_energy_b = float(baseline_df['Energy'].mean())
                                    # preferred_label, preferred_result = select_model_by_rmse_r2(three_res, five_res, rel_tol_pct,mean_kwh)

                                    # -------------------------
                                    # RUN MODELS FOR REPORTED
                                    # -------------------------
                                    temp_r = reported_df['temperature'].dropna().values
                                    energy_r = reported_df['Energy'].dropna().values

                                    if len(temp_r) == len(energy_r):

                                        with st.spinner('Running change-point models...'):
                                            three_res_r = None
                                            five_res_r = None

                                            if model_choice == '3-parameter':
                                                three_res_r = fit_three_param_cp(temp_r, energy_r, Tmin_r, Tmax_r, step,
                                                                                 mode=mode)

                                            if model_choice == '5-parameter':
                                                five_res_r = fit_five_param_deadband(temp_r, energy_r, Tmin_r, Tmax_r, step)

                                            if model_choice == 'Both':
                                                three_res_r = fit_three_param_cp(temp_r, energy_r, Tmin_r, Tmax_r, step,
                                                                                 mode=mode)
                                                five_res_r = fit_five_param_deadband(temp_r, energy_r, Tmin_r, Tmax_r, step)

                                        mean_energy_r = float(reported_df['Energy'].mean())
                                        # preferred_label, preferred_result = select_model_by_rmse_r2(three_res, five_res, rel_tol_pct,mean_kwh)


                                    # ----------------------------
                                    # DISPLAY RESULTS FOR BASELINE
                                    # ----------------------------
                                    st.write("## Model Results")

                                    if model_choice in ["3-parameter"]:
                                        three_para_results(three_res_b, temperature_unit, mean_energy_b)

                                    if model_choice in ["5-parameter"]:
                                        five_para_results(five_res_b, temperature_unit, mean_energy_b)

                                    if model_choice in ["Both"]:
                                        three_five_para_results(three_res_b, five_res_b, temperature_unit, mean_energy_b)


                                    # ----------------------------
                                    # DISPLAY RESULTS FOR REPORTED
                                    # ----------------------------
                                    st.write("## Model Results")

                                    if model_choice in ["3-parameter"]:
                                        three_para_results(three_res_r, temperature_unit, mean_energy_r)

                                    if model_choice in ["5-parameter"]:
                                        five_para_results(five_res_r, temperature_unit, mean_energy_r)

                                    if model_choice in ["Both"]:
                                        three_five_para_results(three_res_r, five_res_r, temperature_unit,
                                                                mean_energy_r)


                                    # -------------------------
                                    # PLOT MODELS FOR BASELINE
                                    # -------------------------
                                    T_plot = np.linspace(baseline_df['temperature'].min(), baseline_df['temperature'].max(), 400)

                                    fig, ax = plt.subplots(figsize=(9, 5))
                                    ax.scatter(baseline_df['temperature'], baseline_df['Energy'], label="Measured Energy", s=50)

                                    if model_choice == "3-parameter":
                                        Y3_plot = predict_3p_for_plot(T_plot, three_res_b["Tb"], three_res_b["model"],
                                                                      mode=three_res_b["mode"])
                                        ax.plot(T_plot, Y3_plot, label="3-parameter", linewidth=2.5)

                                    elif model_choice == "5-parameter":
                                        Y5_plot = predict_5p_for_plot(T_plot, five_res_b["Tb_low"], five_res_b["Tb_high"],
                                                                      five_res_b["model"])
                                        ax.plot(T_plot, Y5_plot, label="5-parameter", linewidth=2.5)

                                    else:  # Both
                                        Y3_plot = predict_3p_for_plot(T_plot, three_res_b["Tb"], three_res_b["model"],
                                                                      mode=three_res_b["mode"])
                                        Y5_plot = predict_5p_for_plot(T_plot, five_res_b["Tb_low"], five_res_b["Tb_high"],
                                                                      five_res_b["model"])
                                        ax.plot(T_plot, Y3_plot, label="3-parameter", linewidth=2.5)
                                        ax.plot(T_plot, Y5_plot, label="5-parameter", linewidth=2.5)

                                    # Deadband shade
                                    if model_choice in ["5-parameter", "Both"]:
                                        ax.axvspan(five_res_b["Tb_low"], five_res_b["Tb_high"], alpha=0.08, color="gray",
                                                   label="Deadband")

                                    if temperature_unit == 'celsius':
                                        ax.set_xlabel("Temperature (°C)")
                                    else:
                                        ax.set_xlabel("Temperature (°F)")
                                    ax.set_ylabel("Energy")

                                    if model_choice in ["3-parameter"]:
                                        ax.set_title("3-Parameter Change-Point Model")
                                    elif model_choice in ["5-parameter"]:
                                        ax.set_title("5-Parameter Change-Point Model")
                                    else:
                                        ax.set_title("3-Parameter and 5-parameter Change-Point Models")
                                    ax.legend()
                                    ax.grid(True)

                                    st.pyplot(fig)

                                    # -------------------------
                                    # PLOT MODELS FOR REPORTED
                                    # -------------------------
                                    T_plot = np.linspace(reported_df['temperature'].min(), reported_df['temperature'].max(),
                                                         400)

                                    fig, ax = plt.subplots(figsize=(9, 5))
                                    ax.scatter(reported_df['temperature'], reported_df['Energy'], label="Measured Energy",
                                               s=50)

                                    if model_choice == "3-parameter":
                                        Y3_plot = predict_3p_for_plot(T_plot, three_res_r["Tb"], three_res_r["model"],
                                                                      mode=three_res_r["mode"])
                                        ax.plot(T_plot, Y3_plot, label="3-parameter", linewidth=2.5)

                                    elif model_choice == "5-parameter":
                                        Y5_plot = predict_5p_for_plot(T_plot, five_res_r["Tb_low"], five_res_r["Tb_high"],
                                                                      five_res_r["model"])
                                        ax.plot(T_plot, Y5_plot, label="5-parameter", linewidth=2.5)

                                    else:  # Both
                                        Y3_plot = predict_3p_for_plot(T_plot, three_res_r["Tb"], three_res_r["model"],
                                                                      mode=three_res_r["mode"])
                                        Y5_plot = predict_5p_for_plot(T_plot, five_res_b["Tb_low"], five_res_b["Tb_high"],
                                                                      five_res_r["model"])
                                        ax.plot(T_plot, Y3_plot, label="3-parameter", linewidth=2.5)
                                        ax.plot(T_plot, Y5_plot, label="5-parameter", linewidth=2.5)

                                    # Deadband shade
                                    if model_choice in ["5-parameter", "Both"]:
                                        ax.axvspan(five_res_r["Tb_low"], five_res_r["Tb_high"], alpha=0.08, color="gray",
                                                   label="Deadband")

                                    if temperature_unit == 'celsius':
                                        ax.set_xlabel("Temperature (°C)")
                                    else:
                                        ax.set_xlabel("Temperature (°F)")
                                    ax.set_ylabel("Energy")

                                    if model_choice in ["3-parameter"]:
                                        ax.set_title("3-Parameter Change-Point Model")
                                    elif model_choice in ["5-parameter"]:
                                        ax.set_title("5-Parameter Change-Point Model")
                                    else:
                                        ax.set_title("3-Parameter and 5-parameter Change-Point Models")
                                    ax.legend()
                                    ax.grid(True)

                                    st.pyplot(fig)

                                    # --------------------------

                                    # TMY Calculations--------------------------

                                    x_tmy = avg_weather_tmy['temperature']

                                    if model_choice == "3-parameter":
                                        pred_base_tmy = predict_3p_for_plot(x_tmy.to_numpy(), three_res_b["Tb"],
                                                                            three_res_b["model"],
                                                                            mode=three_res_b["mode"])
                                        pred_rep_tmy = predict_3p_for_plot(x_tmy.to_numpy(), three_res_r["Tb"],
                                                                           three_res_r["model"],
                                                                           mode=three_res_r["mode"])

                                    elif model_choice == "5-parameter":
                                        pred_base_tmy = predict_5p_for_plot(x_tmy.to_numpy(), five_res_b["Tb_low"],
                                                                            five_res_b["Tb_high"],
                                                                            five_res_b["model"])
                                        pred_rep_tmy = predict_5p_for_plot(x_tmy.to_numpy(), five_res_r["Tb_low"],
                                                                           five_res_r["Tb_high"],
                                                                           five_res_r["model"])

                                    else:  # Both
                                        pred_base_tmy_3p = predict_3p_for_plot(x_tmy.to_numpy(), three_res_b["Tb"],
                                                                               three_res_b["model"],
                                                                               mode=three_res_b["mode"])
                                        pred_rep_tmy_3p = predict_3p_for_plot(x_tmy.to_numpy(), three_res_r["Tb"],
                                                                              three_res_r["model"],
                                                                              mode=three_res_r["mode"])
                                        pred_base_tmy_5p = predict_5p_for_plot(x_tmy.to_numpy(), five_res_b["Tb_low"],
                                                                               five_res_b["Tb_high"],
                                                                               five_res_b["model"])
                                        pred_rep_tmy_5p = predict_5p_for_plot(x_tmy.to_numpy(), five_res_r["Tb_low"],
                                                                              five_res_r["Tb_high"],
                                                                              five_res_r["model"])

                                    if model_choice == "3-parameter" or model_choice == "5-parameter":
                                        st.write(f'##### Predicted Baseline Consumption: \n {pred_base_tmy.sum():.2f}')
                                        st.write(f'##### Predicted Reported Consumption: \n {pred_rep_tmy.sum():.2f}')

                                    else:
                                        mod1, mod2 = st.columns(2, border=True)
                                        with mod1:
                                            st.write(
                                                f'##### 3 Parameter Predicted Baseline Consumption: \n {pred_base_tmy_3p.sum():.2f}')
                                            st.write(
                                                f'##### 3 Parameter Predicted Reported Consumption: \n {pred_rep_tmy_3p.sum():.2f}')
                                        with mod2:
                                            st.write(
                                                f'##### 5 Parameter Predicted Baseline Consumption: \n {pred_base_tmy_5p.sum():.2f}')
                                            st.write(
                                                f'##### 5 Parameter Predicted Reported Consumption: \n {pred_rep_tmy_5p.sum():.2f}')

                                    if model_choice == "3-parameter" or model_choice == "5-parameter":
                                        savings = pred_base_tmy.sum() - pred_rep_tmy.sum()
                                        st.write(f'##### Savings: \n {savings:.2f}')
                                    else:
                                        mod1, mod2 = st.columns(2, border=True)

                                        with mod1:
                                            savings_3p = pred_base_tmy_3p.sum() - pred_rep_tmy_3p.sum()
                                            st.write(f'##### 3 Parameter Savings: \n {savings_3p:.2f}')
                                        with mod2:
                                            savings_5p = pred_base_tmy_5p.sum() - pred_rep_tmy_5p.sum()
                                            st.write(f'##### 5 Parameter Savings: \n {savings_5p:.2f}')

                                else:
                                    st.error('Please select correct interval as per your data.')

                if tmy == 'No':

                    if st.button("Fetch Weather Data & Calculate savings"):
                        if start_date_b > end_date_b and start_date_r > end_date_r:
                            st.error("Start must be <= end")
                        else:
                            start_str_b = start_date_b.isoformat()
                            end_str_b = end_date_b.isoformat()
                            start_str_r = start_date_r.isoformat()
                            end_str_r = end_date_r.isoformat()
                            with st.spinner("Calculating..."):
                                try:
                                    meta, df_weather = fetch_openmeteo_archive(client, lat, lon, start_str_b, end_str_b, temperature_unit, which, var)
                                    meta, df_weather_r = fetch_openmeteo_archive(client, lat, lon, start_str_r, end_str_r, temperature_unit, which, var)
                                except Exception as e:
                                    st.error(f"Weather fetch failed: {e}")
                                else:
                                    st.success("Fetched weather data")

                                    df_temp = df_weather.copy()
                                    df_temp_r = df_weather_r.copy()

                                    if weather_interval == "Hourly":

                                        df_weather_final = df_temp
                                        df_weather_final_r = df_temp_r

                                    elif weather_interval == "Monthly":

                                        df_temp['month'] = df_temp['date_local'].dt.month  # 1–12
                                        df_temp_r['month'] = df_temp_r['date_local'].dt.month

                                        df_weather_final = (df_temp.groupby('month', as_index=False).mean(numeric_only=True))
                                        df_weather_final_r = (df_temp_r.groupby('month', as_index=False).mean(numeric_only=True))

                                    else:

                                        df_temp['month'] = df_temp['date_local'].dt.month
                                        df_temp['day'] = df_temp['date_local'].dt.day
                                        df_temp_r['month'] = df_temp_r['date_local'].dt.month
                                        df_temp_r['day'] = df_temp_r['date_local'].dt.day

                                        df_weather_final = (
                                            df_temp.groupby(['month', 'day'], as_index=False).mean(numeric_only=True))
                                        df_weather_final_r = (
                                            df_temp_r.groupby(['month', 'day'], as_index=False).mean(numeric_only=True))


                                baseline_df = pd.concat([manual_df, df_weather_final], axis=1)
                                reported_df = pd.concat([reported_df, df_weather_final_r], axis=1)

                                # -------------------------
                                # DEFAULT MODEL SETTINGS
                                # -------------------------

                                Tmin_b = float(np.floor(baseline_df['temperature'].min()))
                                Tmax_b = float(np.ceil(baseline_df['temperature'].max()))

                                # -------------------------
                                # RUN MODELS FOR BASELINE
                                # -------------------------
                                temp_b = baseline_df['temperature'].dropna().values
                                energy_b = baseline_df['Energy'].dropna().values

                                if len(temp_b) == len(energy_b):

                                    with st.spinner('Running change-point models...'):
                                        three_res_b = None
                                        five_res_b = None

                                        if model_choice == '3-parameter':
                                            three_res_b = fit_three_param_cp(temp_b, energy_b, Tmin_b, Tmax_b, step, mode=mode)

                                        if model_choice == '5-parameter':
                                            five_res_b = fit_five_param_deadband(temp_b, energy_b, Tmin_b, Tmax_b, step)

                                        if model_choice == 'Both':
                                            three_res_b = fit_three_param_cp(temp_b, energy_b, Tmin_b, Tmax_b, step, mode=mode)
                                            five_res_b = fit_five_param_deadband(temp_b, energy_b, Tmin_b, Tmax_b, step)

                                    mean_energy_b = float(baseline_df['Energy'].mean())
                                    # preferred_label, preferred_result = select_model_by_rmse_r2(three_res, five_res, rel_tol_pct,mean_kwh)

                                    # ----------------------------
                                    # DISPLAY RESULTS FOR BASELINE
                                    # ----------------------------
                                    st.write("## Model Results")

                                    if model_choice in ["3-parameter"]:
                                        three_para_results(three_res_b, temperature_unit, mean_energy_b)

                                    if model_choice in ["5-parameter"]:
                                        five_para_results(five_res_b, temperature_unit, mean_energy_b)

                                    if model_choice in ["Both"]:
                                        three_five_para_results(three_res_b, five_res_b, temperature_unit,
                                                                mean_energy_b)

                                    # -------------------------
                                    # PLOT MODELS FOR BASELINE
                                    # -------------------------
                                    T_plot = np.linspace(baseline_df['temperature'].min(),
                                                         baseline_df['temperature'].max(), 400)

                                    fig, ax = plt.subplots(figsize=(9, 5))
                                    ax.scatter(baseline_df['temperature'], baseline_df['Energy'],
                                               label="Measured Energy", s=50)

                                    if model_choice == "3-parameter":
                                        Y3_plot = predict_3p_for_plot(T_plot, three_res_b["Tb"], three_res_b["model"],
                                                                      mode=three_res_b["mode"])
                                        ax.plot(T_plot, Y3_plot, label="3-parameter", linewidth=2.5)

                                    elif model_choice == "5-parameter":
                                        Y5_plot = predict_5p_for_plot(T_plot, five_res_b["Tb_low"],
                                                                      five_res_b["Tb_high"],
                                                                      five_res_b["model"])
                                        ax.plot(T_plot, Y5_plot, label="5-parameter", linewidth=2.5)

                                    else:  # Both
                                        Y3_plot = predict_3p_for_plot(T_plot, three_res_b["Tb"], three_res_b["model"],
                                                                      mode=three_res_b["mode"])
                                        Y5_plot = predict_5p_for_plot(T_plot, five_res_b["Tb_low"],
                                                                      five_res_b["Tb_high"],
                                                                      five_res_b["model"])
                                        ax.plot(T_plot, Y3_plot, label="3-parameter", linewidth=2.5)
                                        ax.plot(T_plot, Y5_plot, label="5-parameter", linewidth=2.5)

                                    # Deadband shade
                                    if model_choice in ["5-parameter", "Both"]:
                                        ax.axvspan(five_res_b["Tb_low"], five_res_b["Tb_high"], alpha=0.08,
                                                   color="gray",
                                                   label="Deadband")

                                    if temperature_unit == 'celsius':
                                        ax.set_xlabel("Temperature (°C)")
                                    else:
                                        ax.set_xlabel("Temperature (°F)")
                                    ax.set_ylabel("Energy")

                                    if model_choice in ["3-parameter"]:
                                        ax.set_title("3-Parameter Change-Point Model")
                                    elif model_choice in ["5-parameter"]:
                                        ax.set_title("5-Parameter Change-Point Model")
                                    else:
                                        ax.set_title("3-Parameter and 5-parameter Change-Point Models")
                                    ax.legend()
                                    ax.grid(True)

                                    st.pyplot(fig)


                                    # --------------------------

                                    y_r = reported_df['Energy']
                                    x_r = reported_df['temperature']

                                    if model_choice == "3-parameter":
                                        pred_r = predict_3p_for_plot(x_r.to_numpy(), three_res_b["Tb"],
                                                                     three_res_b["model"],
                                                                     mode=three_res_b["mode"])

                                    elif model_choice == "5-parameter":
                                        pred_r = predict_5p_for_plot(x_r.to_numpy(), five_res_b["Tb_low"],
                                                                     five_res_b["Tb_high"],
                                                                     five_res_b["model"])

                                    else:  # Both
                                        pred_r_3p = predict_3p_for_plot(x_r.to_numpy(), three_res_b["Tb"],
                                                                        three_res_b["model"],
                                                                        mode=three_res_b["mode"])
                                        pred_r_5p = predict_5p_for_plot(x_r.to_numpy(), five_res_b["Tb_low"],
                                                                        five_res_b["Tb_high"],
                                                                        five_res_b["model"])

                                    if model_choice == "3-parameter" or model_choice == "5-parameter":
                                        st.write(f'##### Predicted Baseline Consumption: \n {pred_r.sum():.2f}')

                                    else:
                                        mod1, mod2 = st.columns(2, border=True)
                                        with mod1:
                                            st.write(
                                                f'##### 3 Parameter Predicted Baseline Consumption: \n {pred_r_3p.sum():.2f}')
                                        with mod2:
                                            st.write(
                                                f'##### 5 Parameter Predicted Baseline Consumption: \n {pred_r_5p.sum():.2f}')

                                    st.write(f'##### Reported Consumption: \n {y_r.sum():.2f}')

                                    if model_choice == "3-parameter" or model_choice == "5-parameter":
                                        savings = pred_r.sum() - y_r.sum()
                                        st.write(f'##### Savings: \n {savings:.2f}')
                                    else:
                                        mod1, mod2 = st.columns(2, border=True)

                                        with mod1:
                                            savings_3p = pred_r_3p.sum() - y_r.sum()
                                            st.write(f'##### 3 Parameter Savings: \n {savings_3p:.2f}')
                                        with mod2:
                                            savings_5p = pred_r_5p.sum() - y_r.sum()
                                            st.write(f'##### 5 Parameter Savings: \n {savings_5p:.2f}')


                                else:
                                    st.error('Please select correct interval as per your data.')