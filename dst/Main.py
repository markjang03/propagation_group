import streamlit as st
import pandas as pd
import numpy as np
import os
import scipy.io as sio
from common import *
from PIL import Image
from utils import *
from plotter import *

st.set_page_config(
    page_title="Digital Spectrum Twins",
    page_icon=":world_map:",
)


add_logo()


st.write("# Digital Spectrum Twinning for Enhanced Spectrum Sharing and Other Radio Applications")

st.sidebar.success("Select a use case above.")


image = Image.open("ExampleDST_st.png")

st.image(image, caption='Sunrise by the mountains')