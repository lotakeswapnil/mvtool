# import necessary packages
import streamlit as st

st.title('Energy M&V Tool')

# --- Select Mode of Analysis ---

mode = st.radio('Select any one Option:',options=["Manual Data", "Upload Data (CSV or Excel)"], index=None)

with st.spinner('Loading...'):

# -------------------------
# UPLOAD DATA MODE
# -------------------------

    if mode == "Upload Data (CSV or Excel)":
        from supplementary.upload import upload_page
        upload_page()

# -------------------------
# MANUAL DATA MODE
# -------------------------

    elif mode == "Manual Data":
        from supplementary.manual import manual_page
        manual_page()
