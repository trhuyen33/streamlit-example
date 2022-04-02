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

with features:
    st.header('The features I created')
    st.text('I found dataset on...')

with model_training:
    st.header('Time to train model')
    st.text('I found dataset on...')