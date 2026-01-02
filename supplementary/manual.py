# import necessary packages
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import root_mean_squared_error

from supplementary.change_point import (fit_three_param_cp, fit_five_param_deadband, predict_3p_for_plot,predict_5p_for_plot)
from supplementary.weather import make_openmeteo_client, fetch_openmeteo_archive


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


            else:
                st.info('Please complete all Independent Variable names.')


        if weather_data == 'Yes':

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

            Tmin = float(np.floor(final_df['Temperature'].min()))
            Tmax = float(np.ceil(final_df['Temperature'].max()))
            step = 1.0
            rel_tol_pct = 0.1  # 0.1% RMSE tie tolerance


            # -------------------------
            # RUN MODELS
            # -------------------------
            temp = final_df['Temperature'].values
            energy = final_df['Energy'].values

            temp_sel,mod1,mod2 = st.columns([0.25,0.25,0.25])

            with temp_sel:
                temperature_unit = st.selectbox('Select Temperature unit:',['celsius','fahrenheit'])

            with mod1:
                model_choice = st.selectbox("Select Change-Point Model:",["3-parameter", "5-parameter", "Both"])

            with mod2:
                if model_choice == "3-parameter":
                    mode = st.selectbox("Select Change-Point Model Type:",["auto", "heating", "cooling"],index=0)
                elif model_choice == "5-parameter":
                    # Disable the mode selection if the model is not "3-parameter"
                    mode_disabled = model_choice != "3-parameter"

                    mode = st.selectbox("Select Change-Point Model Type:",["auto", "heating", "cooling"],
                        index=0, disabled=mode_disabled)
                else:
                    mode = st.selectbox("Select Change-Point Model Type:",["auto", "heating", "cooling"],index=0)


            if st.button("Run Regression Models"):
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
                # EQUATION DISPLAY
                # -------------------------

                st.write("## Model Equations")

                if model_choice in ["3-parameter", "Both"]:
                    st.write('### 3-parameter:')
                    Tb = three_res["Tb"]
                    b0 = three_res["model"].intercept_
                    b1 = three_res["model"].coef_[0]
                    mode_used = three_res["mode"]  # "heating" or "cooling"

                    if mode_used == "cooling":
                        # Cooling: Energy = b0 + b1 * max(0, T - Tb)
                        st.latex(
                            fr"\text{{Energy}} = {b0:.2f} + {b1:.2f}\,\max(0,\,T - {Tb:.2f})"
                        )

                    elif mode_used == "heating":
                        # Heating: Energy = b0 + b1 * max(0, Tb - T)
                        st.latex(
                            fr"\text{{Energy}} = {b0:.2f} + {b1:.2f}\,\max(0,\,{Tb:.2f} - T)"
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
                    if temperature_unit == 'celsius':
                        st.write(f"**Tb:** {three_res['Tb']:.2f} °C")
                    else:
                        st.write(f"**Tb:** {three_res['Tb']:.2f} °F")
                    st.write(f"**β0:** {three_res['model'].intercept_:.2f}")
                    st.write(f"**β1:** {three_res['model'].coef_[0]:.2f}")
                    st.write(f"**RMSE:** {three_res['rmse']:.2f}")
                    st.write(f"**R²:** {three_res['r2']:.2f}")

                if model_choice in ["5-parameter"]:
                    st.subheader("5-Parameter Model")
                    if temperature_unit == 'celsius':
                        st.write(f"**Tb_low:** {five_res['Tb_low']:.2f} °C")
                    else:
                        st.write(f"**Tb_low:** {five_res['Tb_low']:.2f} °F")
                    if temperature_unit == 'celsius':
                        st.write(f"**Tb_high:** {five_res['Tb_high']:.2f} °C")
                    else:
                        st.write(f"**Tb_high:** {five_res['Tb_high']:.2f} °F")
                    st.write(f"**β0:** {five_res['model'].intercept_:.2f}")
                    st.write(f"**β_h:** {five_res['model'].coef_[0]:.2f}")
                    st.write(f"**β_c:** {five_res['model'].coef_[1]:.2f}")
                    st.write(f"**RMSE:** {five_res['rmse']:.2f}")
                    st.write(f"**R²:** {five_res['r2']:.2f}")

                if model_choice in ["Both"]:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("3-Parameter Model")
                        if temperature_unit == 'celsius':
                            st.write(f"**Tb:** {three_res['Tb']:.2f} °C")
                        else:
                            st.write(f"**Tb:** {three_res['Tb']:.2f} °F")
                        st.write(f"**β0:** {three_res['model'].intercept_:.2f}")
                        st.write(f"**β1:** {three_res['model'].coef_[0]:.2f}")
                        st.write(f"**RMSE:** {three_res['rmse']:.2f}")
                        st.write(f"**R²:** {three_res['r2']:.2f}")
                    with col2:
                        st.subheader("5-Parameter Model")
                        if temperature_unit == 'celsius':
                            st.write(f"**Tb_low:** {five_res['Tb_low']:.2f} °C")
                        else:
                            st.write(f"**Tb_low:** {five_res['Tb_low']:.2f} °F")
                        if temperature_unit == 'celsius':
                            st.write(f"**Tb_high:** {five_res['Tb_high']:.2f} °C")
                        else:
                            st.write(f"**Tb_high:** {five_res['Tb_high']:.2f} °F")
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

    elif manual_data == 'Yes':

        interval = st.radio('Do you want Weather Data using Intervals?', options=["Yes", "No"], index=None, key='interval_data')

        if interval == 'Yes':

            lat, lon = st.columns(2)

            with lat:
                lat = st.number_input("Latitude", format="%.4f")

            with lon:
                lon = st.number_input("Longitude", format="%.4f")


            var = "temperature"  # or let user pick
            which = "hourly"


            # create client once (you can cache it)

            client = make_openmeteo_client()

            temp, step_unit, model_c, model_m = st.columns(4)

            with temp:
                temperature_unit = st.selectbox('Select Temperature Unit:',['celsius','fahrenheit'])

            with step_unit:
                step = st.selectbox('Select Decimals For Balance Point:',[1.0,0.5,0.1])

            with model_c:
                model_choice = st.selectbox("Select Change-Point Model:", ["3-parameter", "5-parameter", "Both"])

            with model_m:
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

            # Build column names automatically
            col_names = ['Start Date','End Date','Energy']  # first column fixed
            empty_df = pd.DataFrame({'Start Date (yyyy-mm-dd)': pd.Series(['2025-01-01'], dtype='datetime64[ns]'),
                                     'End Date (yyyy-mm-dd)': pd.Series(['2025-02-01'], dtype='datetime64[ns]'),
                                     'Energy': pd.Series([0], dtype=float)})

            st.write('#### Enter Baseline Energy Data Below:')

            final_df = st.data_editor(empty_df, num_rows="dynamic", key='baseline_energy')

            if len(final_df) < 2:
                st.error("Please enter at least 2 rows.")

            st.write('#### Enter Reported Energy Data Below:')
            reported_df = st.data_editor(empty_df, num_rows="dynamic", key='reported_energy')

            if len(reported_df) < 2:
                st.error("Please enter at least 2 rows.")

            if st.button("Fetch Weather Data & Run Regression"):

                with st.spinner('Calculating...'):
                
                    for i in range(len(final_df)):
                        start_date = final_df['Start Date (yyyy-mm-dd)'][i].date().isoformat()
                        end_date = final_df['End Date (yyyy-mm-dd)'][i].date().isoformat()
                        meta, temperature_data = fetch_openmeteo_archive(client, lat, lon, start_date, end_date, temperature_unit, which, var)
                        final_df.loc[i, 'temperature'] = temperature_data["temperature"].mean()
                        # st.write(manual_df)
    
                    # -------------------------
                    # DEFAULT MODEL SETTINGS
                    # -------------------------
    
                    Tmin = float(np.floor(final_df['temperature'].min()))
                    Tmax = float(np.ceil(final_df['temperature'].max()))
                    rel_tol_pct = 0.1  # 0.1% RMSE tie tolerance
    
                    # -------------------------
                    # RUN MODELS
                    # -------------------------
                    temp = final_df['temperature'].values
                    energy = final_df['Energy'].values
    
                    three_res = None
                    five_res = None
    
                    if model_choice == "3-parameter":
                        three_res = fit_three_param_cp(temp, energy, Tmin, Tmax, step, mode=mode)
    
                    if model_choice == "5-parameter":
                        five_res = fit_five_param_deadband(temp, energy, Tmin, Tmax, step)
    
                    if model_choice == "Both":
                        three_res = fit_three_param_cp(temp, energy, Tmin, Tmax, step, mode=mode)
                        five_res = fit_five_param_deadband(temp, energy, Tmin, Tmax, step)
    
                    mean_energy = float(final_df['Energy'].mean())
                    # preferred_label, preferred_result = select_model_by_rmse_r2(three_res, five_res, rel_tol_pct,mean_energy)
    
                    # -------------------------
                    # EQUATION DISPLAY
                    # -------------------------
                    st.write("## Model Equations")
    
                    if model_choice in ["3-parameter", "Both"]:
                        st.write('### 3-parameter:')
                        Tb = three_res["Tb"]
                        b0 = three_res["model"].intercept_
                        b1 = three_res["model"].coef_[0]
                        mode_used = three_res["mode"]  # "heating" or "cooling"
    
                        if mode_used == "cooling":
                            # Cooling: Energy = b0 + b1 * max(0, T - Tb)
                            st.latex(
                                fr"\text{{Energy}} = {b0:.2f} + {b1:.2f}\,\max(0,\,T - {Tb:.2f})"
                            )
    
                        elif mode_used == "heating":
                            # Heating: Energy = b0 + b1 * max(0, Tb - T)
                            st.latex(
                                fr"\text{{Energy}} = {b0:.2f} + {b1:.2f}\,\max(0,\,{Tb:.2f} - T)"
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
                        if temperature_unit == 'celsius':
                            st.write(f"**Tb:** {three_res['Tb']:.2f} °C")
                        else:
                            st.write(f"**Tb:** {three_res['Tb']:.2f} °F")
                        st.write(f"**β0:** {three_res['model'].intercept_:.2f}")
                        st.write(f"**β1:** {three_res['model'].coef_[0]:.2f}")
                        st.write(f"**RMSE:** {three_res['rmse']:.2f}")
                        st.write(f"**R²:** {three_res['r2']:.2f}")
    
                    if model_choice in ["5-parameter"]:
                        st.subheader("5-Parameter Model")
                        if temperature_unit == 'celsius':
                            st.write(f"**Tb_low:** {five_res['Tb_low']:.2f} °C")
                        else:
                            st.write(f"**Tb_low:** {five_res['Tb_low']:.2f} °F")
                        if temperature_unit == 'celsius':
                            st.write(f"**Tb_high:** {five_res['Tb_high']:.2f} °C")
                        else:
                            st.write(f"**Tb_high:** {five_res['Tb_high']:.2f} °F")
                        st.write(f"**β0:** {five_res['model'].intercept_:.2f}")
                        st.write(f"**β_h:** {five_res['model'].coef_[0]:.2f}")
                        st.write(f"**β_c:** {five_res['model'].coef_[1]:.2f}")
                        st.write(f"**RMSE:** {five_res['rmse']:.2f}")
                        st.write(f"**R²:** {five_res['r2']:.2f}")
    
                    if model_choice in ["Both"]:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("3-Parameter Model")
                            if temperature_unit == 'celsius':
                                st.write(f"**Tb:** {three_res['Tb']:.2f} °C")
                            else:
                                st.write(f"**Tb:** {three_res['Tb']:.2f} °F")
                            st.write(f"**β0:** {three_res['model'].intercept_:.2f}")
                            st.write(f"**β1:** {three_res['model'].coef_[0]:.2f}")
                            st.write(f"**RMSE:** {three_res['rmse']:.2f}")
                            st.write(f"**R²:** {three_res['r2']:.2f}")
                        with col2:
                            st.subheader("5-Parameter Model")
                            if temperature_unit == 'celsius':
                                st.write(f"**Tb_low:** {five_res['Tb_low']:.2f} °C")
                            else:
                                st.write(f"**Tb_low:** {five_res['Tb_low']:.2f} °F")
                            if temperature_unit == 'celsius':
                                st.write(f"**Tb_high:** {five_res['Tb_high']:.2f} °C")
                            else:
                                st.write(f"**Tb_high:** {five_res['Tb_high']:.2f} °F")
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


        if interval == 'No':


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
                start_date = st.date_input("Start date", value=date.today()- timedelta(days=365))

            with lon:
                lon = st.number_input("Longitude", format="%.4f")
                end_date = st.date_input("End date", value=date.today()- timedelta(days=364))

            var = "temperature"  # or let user pick
            which = "hourly"


            # create client once (you can cache it)

            client = make_openmeteo_client()

            weather_i, model_c = st.columns(2)

            with weather_i:
                weather_interval = st.selectbox('Select Interval', {'Hourly', 'Daily', 'Monthly'})
                model_choice = st.selectbox("Select Change-Point Model:", ["3-parameter", "5-parameter", "Both"])

            with model_c:
                temperature_unit = st.selectbox('Select Temperature Unit:', ['celsius', 'fahrenheit'])
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

            if st.button("Fetch Weather Data & Run Regression"):
                if start_date > end_date:
                    st.error("Start must be <= end")
                else:
                    start_str = start_date.isoformat()
                    end_str = end_date.isoformat()
                    with st.spinner("Fetching..."):
                        try:
                            meta, df_weather = fetch_openmeteo_archive(client, lat, lon, start_str, end_str, temperature_unit, which, var)
                        except Exception as e:
                            st.error(f"Weather fetch failed: {e}")
                        else:
                            st.success("Fetched weather data")

                            df_temp = df_weather.copy()

                            if weather_interval == "Hourly":

                                df_weather_final = df_temp

                            elif weather_interval == "Monthly":

                                df_temp['month'] = df_temp['date_local'].dt.month  # 1–12

                                df_weather_final = (df_temp.groupby('month', as_index=False).mean(numeric_only=True))

                            else:

                                df_temp['month'] = df_temp['date_local'].dt.month
                                df_temp['day'] = df_temp['date_local'].dt.day

                                df_weather_final = (
                                    df_temp.groupby(['month', 'day'], as_index=False).mean(numeric_only=True))


                            final_df = pd.concat([manual_df, df_weather_final], axis=1)

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
                            temp = final_df['temperature'].dropna().values
                            energy = final_df['Energy'].dropna().values

                            if len(temp) == len(energy):

                                with st.spinner('Running change-point models...'):
                                    three_res = None
                                    five_res = None

                                    if model_choice == '3-parameter':
                                        three_res = fit_three_param_cp(temp, energy, Tmin, Tmax, step, mode=mode)

                                    if model_choice == '5-parameter':
                                        five_res = fit_five_param_deadband(temp, energy, Tmin, Tmax, step)

                                    if model_choice == 'Both':
                                        three_res = fit_three_param_cp(temp, energy, Tmin, Tmax, step, mode=mode)
                                        five_res = fit_five_param_deadband(temp, energy, Tmin, Tmax, step)

                                mean_energy = float(final_df['Energy'].mean())
                                # preferred_label, preferred_result = select_model_by_rmse_r2(three_res, five_res, rel_tol_pct,mean_kwh)

                                # -------------------------
                                # EQUATION DISPLAY
                                # -------------------------
                                st.write("## Model Equations")

                                if model_choice in ['3-parameter', 'Both']:
                                    st.write('### 3-parameter:')
                                    Tb = three_res['Tb']
                                    b0 = three_res['model'].intercept_
                                    b1 = three_res['model'].coef_[0]
                                    mode_used = three_res['mode']  # "heating" or "cooling"

                                    if mode_used == "cooling":
                                        # Cooling: Energy = b0 + b1 * max(0, T - Tb)
                                        st.latex(
                                            fr"\text{{Energy}} = {b0:.2f} + {b1:.2f}\,\max(0,\,T - {Tb:.2f})"
                                        )

                                    elif mode_used == "heating":
                                        # Heating: Energy = b0 + b1 * max(0, Tb - T)
                                        st.latex(
                                            fr"\text{{Energy}} = {b0:.2f} + {b1:.2f}\,\max(0,\,{Tb:.2f} - T)"
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
                                    if temperature_unit == 'celsius':
                                        st.write(f"**Tb:** {three_res['Tb']:.2f} °C")
                                    else:
                                        st.write(f"**Tb:** {three_res['Tb']:.2f} °F")
                                    st.write(f"**β0:** {three_res['model'].intercept_:.2f}")
                                    st.write(f"**β1:** {three_res['model'].coef_[0]:.2f}")
                                    st.write(f"**RMSE:** {three_res['rmse']:.2f}")
                                    st.write(f"**R²:** {three_res['r2']:.2f}")

                                if model_choice in ["5-parameter"]:
                                    st.subheader("5-Parameter Model")
                                    if temperature_unit == 'celsius':
                                        st.write(f"**Tb_low:** {five_res['Tb_low']:.2f} °C")
                                    else:
                                        st.write(f"**Tb_low:** {five_res['Tb_low']:.2f} °F")
                                    if temperature_unit == 'celsius':
                                        st.write(f"**Tb_high:** {five_res['Tb_high']:.2f} °C")
                                    else:
                                        st.write(f"**Tb_high:** {five_res['Tb_high']:.2f} °F")
                                    st.write(f"**β0:** {five_res['model'].intercept_:.2f}")
                                    st.write(f"**β_h:** {five_res['model'].coef_[0]:.2f}")
                                    st.write(f"**β_c:** {five_res['model'].coef_[1]:.2f}")
                                    st.write(f"**RMSE:** {five_res['rmse']:.2f}")
                                    st.write(f"**R²:** {five_res['r2']:.2f}")

                                if model_choice in ["Both"]:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.subheader("3-Parameter Model")
                                        if temperature_unit == 'celsius':
                                            st.write(f"**Tb:** {three_res['Tb']:.2f} °C")
                                        else:
                                            st.write(f"**Tb:** {three_res['Tb']:.2f} °F")
                                        st.write(f"**β0:** {three_res['model'].intercept_:.2f}")
                                        st.write(f"**β1:** {three_res['model'].coef_[0]:.2f}")
                                        st.write(f"**RMSE:** {three_res['rmse']:.2f}")
                                        st.write(f"**R²:** {three_res['r2']:.2f}")
                                    with col2:
                                        st.subheader("5-Parameter Model")
                                        if temperature_unit == 'celsius':
                                            st.write(f"**Tb_low:** {five_res['Tb_low']:.2f} °C")
                                        else:
                                            st.write(f"**Tb_low:** {five_res['Tb_low']:.2f} °F")
                                        if temperature_unit == 'celsius':
                                            st.write(f"**Tb_high:** {five_res['Tb_high']:.2f} °C")
                                        else:
                                            st.write(f"**Tb_high:** {five_res['Tb_high']:.2f} °F")
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

                                if temperature_unit == 'celsius':
                                    ax.set_xlabel("Temperature (°C)")
                                else:
                                    ax.set_xlabel("Temperature (°F)")
                                ax.set_ylabel("Energy")
                                ax.set_title("3-Parameter vs 5-Parameter Change-Point Models")
                                ax.legend()
                                ax.grid(True)

                                st.pyplot(fig)

                            else:
                                st.error('Please select correct interval as per your data.')
