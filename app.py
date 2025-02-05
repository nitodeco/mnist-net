import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import numpy as np
from PIL import Image
from core.model import model, device
from torch import load
from views import train, test, eval

st.set_page_config(
    page_title="MNIST Net",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

views = {
    "Training": train,
    "Evaluation": eval,
    "Testing": test,
}

st.sidebar.title("MNIST Net")
selection = st.sidebar.radio("Navigation", list(views.keys()), key="navigation_radio")

views[selection]()
