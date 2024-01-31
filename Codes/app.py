import sys
import os
# External packages
import streamlit as st

# Local Modules
import helper

sys.path.append(os.path.join(os.path.dirname(__file__),'detect_wrapper'))
sys.path.append(os.path.join(os.path.dirname(__file__),'tracking_wrapper\\dronetracker'))
sys.path.append(os.path.join(os.path.dirname(__file__),'tracking_wrapper\\drtracker'))

# Setting page layout
st.set_page_config(
    page_title="Object Tracking",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Tracking")

st.sidebar.header("Video Object Tracking")
helper.play_stored_video()
