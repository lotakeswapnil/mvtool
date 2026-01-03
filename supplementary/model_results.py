import streamlit as st

def three_para_results(three_res,temperature_unit,mean_energy):
    st.subheader("3-Parameter Model")
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