# import necessary packages
import streamlit as st

st.title('Energy M&V Tool')

st.sidebar.header("Menu")

# --- Create session-state variable ---
if "mode" not in st.session_state:
    st.session_state.mode = None

# --- Display Start Buttons ---

with st.sidebar:
    mode = st.radio(
        "Navigation",
        ["Main Menu", "Manual Entry", "Upload Data"],
        index=0
    )

# Main content
if mode == "Main Menu":
    st.subheader("Select Any One of the Options")

    st.markdown(
        """
        Choose how you want to provide data.
        Use the sidebar to navigate afterward.
        """
    )

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
