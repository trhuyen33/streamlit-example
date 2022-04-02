from email import header
from pyexpat import features
import streamlit as st

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()

with header:
    st.title('Welcome to my project')
    st.text('In this project I look into the...')

with dataset:
    st.header('Dataset')
    st.text('I found dataset on...')