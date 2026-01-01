# import necessary packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import root_mean_squared_error

from supplementary.change_point import (fit_three_param_cp, fit_five_param_deadband, predict_3p_for_plot,predict_5p_for_plot)

def upload_page():

    col1, col2 = st.columns(2)

    with col1:
        st.write('### Baseline Data')
        uploaded_b = st.file_uploader('', type=['csv', 'xlsx', 'xls'], label_visibility='collapsed', key='baseline')

    with col2:
        st.write('### Reported Data')
        uploaded_r = st.file_uploader('', type=['csv', 'xlsx', 'xls'], label_visibility='collapsed', key='reported')

    if uploaded_b:

        if uploaded_b.name.endswith('.csv'):
            df_b = pd.read_csv(uploaded_b)
            st.write('### Baseline Data Preview:', df_b.head())
        else:
            df_b = pd.read_excel(uploaded_b)
            st.write('### Baseline Data Preview:', df_b.head())

        if uploaded_r:
            if uploaded_r.name.endswith('.csv'):
                df_r = pd.read_csv(uploaded_r)
                st.write('### Reported Data Preview:', df_r.head())
            else:
                df_r = pd.read_excel(uploaded_r)
                st.write('### Reported Data Preview:', df_r.head())


        st.subheader('Select Baseline Data details:')

        base1, base2 = st.columns(2)

        with base1:
            base_ind_var = st.selectbox('Select Independent Variable Type', {'Temperature', 'Independent Variable'})

            if base_ind_var == 'Independent Variable':
                # Number of independent vars
                base_num_var = st.number_input('Number of Independent Variables', min_value=1, max_value=10, step=1)

            else:
                temp_data = st.text_input('Temperature column name')


        with base2:
            if base_ind_var == 'Independent Variable':

                # Target (dependent) column
                base_energy = st.text_input('Baseline Energy column name')

                for i in range(1, base_num_var + 1):
                    globals()[f"ind_var_{i}"] = st.text_input(f"Independent Variable {i}",key=f"var_{i}")

            else:
                base_energy = st.text_input('Energy column name')
                temperature_unit = st.selectbox('Select Temperature Unit:', ['celsius', 'fahrenheit'])


        if base_ind_var == 'Independent Variable':

            st.subheader('Reported Data details:')

            rep1, rep2, rep3 = st.columns(3)

            with rep1:
                rep_ind_var = base_ind_var
                st.write(f'###### Reported Variable Name: \n {rep_ind_var}')

            with rep2:
                rep_energy = base_energy
                st.write(f'###### Reported Energy column name: \n {rep_energy}')

            with rep3:
                if rep_ind_var == 'Independent Variable':
                    st.write(f'###### Reported Independent Variable Names:')
                    for i in range(1, base_num_var + 1):
                        st.write(globals()[f"ind_var_{i}"])

                else:
                    st.write(f'###### Reported Independent Variable Name: \n {temp_data}')

            if st.button('Calculate Savings'):
                if base_energy is not None and globals()[f"ind_var_{i}"] != "":
                    if globals()[f"ind_var_{i}"] not in df_b.columns:
                        st.error(f"Variable '{globals()[f'ind_var_{i}']}' not found in the uploaded CSV.")
                    else:
                        # ---------- ADDED: build list of independent variables ----------
                        independent = [
                            globals()[f"ind_var_{j}"]
                            for j in range(1, base_num_var + 1)
                            if globals()[f"ind_var_{j}"] in df_b.columns
                        ]
                        # -----------------------------------------------------------------

                        # ---------- MODIFIED MINIMALLY: use the list instead of single var ----------
                        X = df_b[independent]  # <- works for 1 or many variables
                        # ------------------------------------------------------------------------------

                        y = df_b[base_energy]

                        model = LinearRegression()
                        model.fit(X, y)
                        preds = model.predict(X)
                        regression = model.score(X, y)
                        cvrmse = root_mean_squared_error(y, preds)/y.mean()

                        #--------------------------------------------------
                        y_r = df_r[rep_energy]
                        x_r = df_r[independent]
                        pred_r = model.predict(x_r)
                        savings = pred_r.sum() - df_r[rep_energy].sum()
                        st.write(f'### Savings: {savings:.2f}')

                        # ---------- ADDED: Regression Equation Display ----------
                        coef = model.coef_
                        intercept = model.intercept_

                        equation_latex = (
                                "Energy = " +
                                f"{intercept:.2f} + " +
                                " + ".join([f"{coef[k]:.2f} \\times {independent[k]}" for k in range(len(independent))])
                        )

                        st.subheader("Regression Equation")
                        st.latex(equation_latex)

                        st.write(f'R2: {regression:.2%}')
                        st.write(f'CV (RMSE): {cvrmse:.2%}')

                        if base_num_var == 1:
                            chart_df = pd.DataFrame({independent[0]: X.iloc[:, 0], "Actual": y, "Predicted": preds})
                            st.line_chart(chart_df, x=independent[0], y=["Actual", "Predicted"])
                        else:
                            st.line_chart(pd.DataFrame({'Actual': y, 'Predicted': preds}).reset_index(drop=True))

                else:
                    st.error('All variables not defined.')


        # -------------------------
        # Sample data (heating + deadband + cooling)
        # -------------------------

        if base_ind_var == 'Temperature':

            mod1, mod2 = st.columns([0.25, 0.25])

            with mod1:
                model_choice = st.selectbox("Select Change-Point Model:", ["3-parameter", "5-parameter", "Both"])

            with mod2:
                if model_choice == "3-parameter":
                    mode = st.selectbox("Select Change-Point Model Type:", ["auto", "heating", "cooling"], index=0)
                elif model_choice == "5-parameter":
                    # Disable the mode selection if the model is not "3-parameter"
                    mode_disabled = model_choice != "3-parameter"

                    mode = st.selectbox("Select Change-Point Model Type:", ["auto", "heating", "cooling"],
                                        index=0, disabled=mode_disabled)
                else:
                    mode = st.selectbox("Select Change-Point Model Type:", ["auto", "heating", "cooling"], index=0)

            st.subheader('Reported Data details:')

            rep1, rep2, rep3 = st.columns(3)

            with rep1:
                rep_ind_var = base_ind_var
                st.write(f'###### Reported Variable Name: \n {rep_ind_var}')

            with rep2:
                rep_energy = base_energy
                st.write(f'###### Reported Energy column name: \n {rep_energy}')

            with rep3:
                if rep_ind_var == 'Independent Variable':
                    st.write(f'###### Reported Independent Variable Names:')
                    for i in range(1, base_num_var + 1):
                        st.write(globals()[f"ind_var_{i}"])

                else:
                    st.write(f'###### Reported Independent Variable Name: \n {temp_data}')


            if st.button('Calculate Savings'):
                if temp_data != '' and base_energy != '':

                    # -------------------------
                    # DEFAULT MODEL SETTINGS
                    # -------------------------
                    Tmin = float(np.floor(df_b[temp_data].min()))
                    Tmax = float(np.ceil(df_b[temp_data].max()))
                    step = 1.0
                    rel_tol_pct = 0.1  # 0.1% RMSE tie tolerance



                    # -------------------------
                    # RUN MODELS
                    # -------------------------
                    temp = df_b[temp_data].values
                    energy = df_b[base_energy].values

                    with st.spinner("Running change-point models..."):
                        three_res = None
                        five_res = None

                        if model_choice == "3-parameter":
                            three_res = fit_three_param_cp(temp, energy, Tmin, Tmax, step, mode=mode)

                        if model_choice == "5-parameter":
                            five_res = fit_five_param_deadband(temp, energy, Tmin, Tmax, step)

                        if model_choice == "Both":
                            three_res = fit_three_param_cp(temp, energy, Tmin, Tmax, step, mode=mode)
                            five_res = fit_five_param_deadband(temp, energy, Tmin, Tmax, step)

                    mean_energy = float(df_b[base_energy].mean())
                    #preferred_label, preferred_result = select_model_by_rmse_r2(three_res, five_res, rel_tol_pct, mean_kwh)


                    # -------------------------
                    # EQUATION DISPLAY
                    # -------------------------
                    st.subheader("Model Equations")

                    if model_choice in ["3-parameter", "Both"]:
                        st.write('#### 3-parameter:')
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
                    st.subheader("Model Results")

                    if model_choice in ["3-parameter"]:
                        st.write("### 3-Parameter Model")
                        if temperature_unit == 'celsius':
                            st.write(f"**Tb:** {three_res['Tb']:.2f} °C")
                        else:
                            st.write(f"**Tb:** {three_res['Tb']:.2f} °F")
                        st.write(f"**β0:** {three_res['model'].intercept_:.2f}")
                        st.write(f"**β1:** {three_res['model'].coef_[0]:.2f}")
                        st.write(f"**RMSE:** {three_res['rmse']:.2f}")
                        st.write(f"**R²:** {three_res['r2']:.2f}")

                    if model_choice in ["5-parameter"]:
                        st.write("### 5-Parameter Model")
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
                            st.write("### 3-Parameter Model")
                            if temperature_unit == 'celsius':
                                st.write(f"**Tb:** {three_res['Tb']:.2f} °C")
                            else:
                                st.write(f"**Tb:** {three_res['Tb']:.2f} °F")
                            st.write(f"**β0:** {three_res['model'].intercept_:.2f}")
                            st.write(f"**β1:** {three_res['model'].coef_[0]:.2f}")
                            st.write(f"**RMSE:** {three_res['rmse']:.2f}")
                            st.write(f"**R²:** {three_res['r2']:.2f}")
                        with col2:
                            st.write("### 5-Parameter Model")
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

                    T_plot = np.linspace(df_b[temp_data].min(), df_b[temp_data].max(), 400)

                    fig, ax = plt.subplots(figsize=(9, 5))
                    ax.scatter(df_b[temp_data], df_b[base_energy], label="Measured Energy", s=50)

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
                        ax.axvspan(five_res["Tb_low"], five_res["Tb_high"], alpha=0.08, color="gray", label="Deadband")

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


                    y_r = df_r[rep_energy]
                    x_r = df_r[temp_data]
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
                        mod1, mod2 = st.columns(2)
                        with mod1:
                            st.write(f'##### 3 Parameter Predicted Baseline Consumption: \n {pred_r_3p.sum():.2f}')
                        with mod2:
                            st.write(f'##### 5 Parameter Predicted Baseline Consumption: \n {pred_r_5p.sum():.2f}')

                    st.write(f'##### Reported Consumption: \n {df_r[rep_energy].sum():.2f}')

                    if model_choice == "3-parameter" or model_choice == "5-parameter":
                        savings = pred_r.sum() - df_r[rep_energy].sum()
                        st.write(f'##### Savings: \n {savings:.2f}')
                    else:
                        mod1, mod2 = st.columns(2)

                        with mod1:
                            savings_3p = pred_r_3p.sum() - df_r[rep_energy].sum()
                            st.write(f'##### 3 Parameter Savings: \n {savings_3p:.2f}')
                        with mod2:
                            savings_5p = pred_r_5p.sum() - df_r[rep_energy].sum()
                            st.write(f'##### 5 Parameter Savings: \n {savings_5p:.2f}')

            if temp_data == '':
                st.error("Please add temperature column name.")

            if base_energy == '':
                st.error("Please add energy column name.")