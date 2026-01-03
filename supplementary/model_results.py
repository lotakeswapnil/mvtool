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