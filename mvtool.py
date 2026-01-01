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
        if st.button("Upload Data (CSV or Excel)"):
            st.session_state.mode = "upload"
            st.rerun()

# -------------------------
# UPLOAD DATA MODE
# -------------------------

elif st.session_state.mode == "upload":

    from supplementary.upload import upload_page

    upload_page()

# -------------------------
# MANUAL DATA MODE
# -------------------------

elif st.session_state.mode == "manual":

    from supplementary.manual import manual_page

    manual_page()