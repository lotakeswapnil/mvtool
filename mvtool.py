# import necessary packages
import streamlit as st

st.title('Energy M&V Tool')


# --- Display Start Buttons ---

mode = st.radio('Select Any One of the Options',options=["Manual Data", "Upload Data (CSV or Excel)"], index=None)

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
