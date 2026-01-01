# import necessary packages
import streamlit as st

st.title('Energy M&V Tool')

st.sidebar.header("Menu")

# --- Create session-state variable ---
if "mode" not in st.session_state:
    st.session_state.mode = None

# --- Display Start Buttons ---

mode = st.radio('Select Any One of the Options',options=["Manual Data", "Upload Data (CSV or Excel)"], index=None)

# -------------------------
# UPLOAD DATA MODE
# -------------------------

if mode == "Upload Data":
    with st.spinner("Loading..."):
        from supplementary.upload import upload_page
        upload_page()

# -------------------------
# MANUAL DATA MODE
# -------------------------

elif mode == "Manual Entry":
    with st.spinner("Loading..."):
        from supplementary.manual import manual_page
        manual_page()
