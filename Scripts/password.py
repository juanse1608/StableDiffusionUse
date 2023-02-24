import streamlit as st

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["PASSWORD"] == st.secrets["ACCESS_TO_APP"]:
            st.session_state["PASSWORD_CORRECT"] = True
            del st.session_state["PASSWORD"]  # don't store password
        else:
            st.session_state["PASSWORD_CORRECT"] = False

    if "PASSWORD_CORRECT" not in st.session_state:
        # First run, show input for password.
        st.text_input("Password", type="password", on_change=password_entered, key="PASSWORD")
        return False
    elif not st.session_state["PASSWORD_CORRECT"]:
        # Password not correct, show input + error.
        st.text_input("Password", type="password", on_change=password_entered, key="PASSWORD")
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True