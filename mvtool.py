# import necessary packages
import streamlit as st

st.title('Energy M&V Tool')

st.sidebar.header("Menu")

# --- Create session-state variable ---
if "mode" not in st.session_state:
    st.session_state.mode = None

# --- Display Start Buttons ---

st.subheader("Select Any One of the Options")

mode = st.radio(
    "Choose input method",
    options=["Manual Entry", "Upload Data"],
    index=None
)

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
