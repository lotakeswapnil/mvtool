# import necessary packages
import streamlit as st

st.title('Energy M&V Tool')

st.sidebar.header("Menu")

# --- Create session-state variable ---
if "mode" not in st.session_state:
    st.session_state.mode = None

# --- Display Start Buttons ---

st.subheader("Select Any One of the Options")

st.write("## Choose input method")

mode = st.radio(options=["Manual Entry", "Upload Data"],disabled=False)

# -------------------------
# UPLOAD DATA MODE
# -------------------------

if mode == "Upload Data":
    from supplementary.upload import upload_page
    upload_page()

# -------------------------
# MANUAL DATA MODE
# -------------------------

elif mode == "Manual Entry":
    from supplementary.manual import manual_page
    manual_page()
