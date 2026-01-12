import streamlit as st

def three_para_results(three_res,temperature_unit,mean_energy):
    st.subheader("3-Parameter Model")
    st.write("#### Model Equations")
    Tb = three_res["Tb"]
    b0 = three_res["model"].intercept_
    b1 = three_res["model"].coef_[0]
    b2 = three_res["model"].coef_[1]
    mode_used = three_res["mode"]  # "heating" or "cooling"

    if mode_used == "cooling":
        # Cooling: Energy = b0 + b1 * max(0, T - Tb)
        st.latex(
            fr"\text{{Energy}} = {b0:.2f} + {b1:.2f}\,\max(0,\,T - {Tb:.2f})"
        )

    elif mode_used == "heating":
        # Heating: Energy = b0 + b1 * max(0, Tb - T)
        st.latex(
            fr"\text{{Energy}} = {b0:.2f} + {b2:.2f} + {b1:.2f}\,\max(0,\,{Tb:.2f} - T)"
        )

    st.write("#### Model Results")
    if temperature_unit == 'celsius':
        st.write(f"**Tb:** {three_res['Tb']:.2f} °C")
    else:
        st.write(f"**Tb:** {three_res['Tb']:.2f} °F")
    st.write(f"**β0:** {three_res['model'].intercept_:.2f}")
    st.write(f"**β1:** {three_res['model'].coef_[0]:.2f}")
    st.write(f"**CV (RMSE):** {three_res['rmse']/mean_energy:.2%}")
    st.write(f"**R²:** {three_res['r2']:.2%}")

def five_para_results(five_res,temperature_unit,mean_energy):
    st.subheader("5-Parameter Model")
    st.write("#### Model Equations")
    st.latex(
        fr"\text{{Energy}} = {five_res['model'].intercept_:.2f} + "
        fr"{five_res['model'].coef_[0]:.2f}\,\max(0,\,{five_res['Tb_low']:.2f} - T) + "
        fr"{five_res['model'].coef_[1]:.2f}\,\max(0,\,T - {five_res['Tb_high']:.2f})"
    )
    st.write("#### Model Results")
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
    st.write(f"**CV (RMSE):** {five_res['rmse'] / mean_energy:.2%}")
    st.write(f"**R²:** {five_res['r2']:.2%}")

def three_five_para_results(three_res,five_res,temperature_unit,mean_energy):
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.subheader("3-Parameter Model")
        st.write("#### Model Equations")
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
        st.write("#### Model Results")
        if temperature_unit == 'celsius':
            st.write(f"**Tb:** {three_res['Tb']:.2f} °C")
        else:
            st.write(f"**Tb:** {three_res['Tb']:.2f} °F")
        st.write(f"**β0:** {three_res['model'].intercept_:.2f}")
        st.write(f"**β1:** {three_res['model'].coef_[0]:.2f}")
        st.write(f"**CV (RMSE):** {three_res['rmse'] / mean_energy:.2%}")
        st.write(f"**R²:** {three_res['r2']:.2%}")
    with col2:
        st.subheader("5-Parameter Model")
        st.write("#### Model Equations")
        st.latex(
            fr"\text{{Energy}} = {five_res['model'].intercept_:.2f} + "
            fr"{five_res['model'].coef_[0]:.2f}\,\max(0,\,{five_res['Tb_low']:.2f} - T) + "
            fr"{five_res['model'].coef_[1]:.2f}\,\max(0,\,T - {five_res['Tb_high']:.2f})"
        )
        st.write("#### Model Results")
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
        st.write(f"**CV (RMSE):** {five_res['rmse'] / mean_energy:.2%}")
        st.write(f"**R²:** {five_res['r2']:.2%}")