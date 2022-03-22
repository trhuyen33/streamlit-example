from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
import cv2
from plotly.subplots import make_subplots

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()
test = st.container()

training_data = pd.read_csv("dataset/train.csv")
testing_data = pd.read_csv("dataset/test.csv")
healthy = Image.open('dataset/train_2.jpg')
multiple_diseases = Image.open('dataset/train_1.jpg')
rust = Image.open('dataset/train_3.jpg')
scrab = Image.open('dataset/train_7.jpg')

model = tf.keras.models.load_model('my_model.hdf5')

def import_and_predict(image_data, model):
    size = (128,128)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = (cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC))/255.
    
    img_reshape = img_resize[np.newaxis,...]

    prediction = model.predict(img_reshape)
    
    return prediction

# def prediction(path, prediction):
#     inputImage = cv2.imread(path)
#     fig = make_subplots(rows=1, cols=2)
#     fig.add_trace(go.Image(z=cv2.resize(inputImage, (512, 512))), row=1, col=1)
#     fig.add_trace(go.Bar(x=["Healthy", "Multiple diseases", "Rust", "Scab"], y=prediction), row=1, col=2)
#     fig.update_layout(height=512, width=900, title_text="DenseNet", showlegend=False)

with header:
    st.title('Đề tài xác định bệnh thực vật')
    st.write('Trong đề tài này, nhóm em tập trung vào việc tìm kiếm dữ liệu, huấn luyện mô hình và kiểm thử')

with dataset:
    st.header('Dataset')
    st.write('Dataset sử dụng trong đề tài được lấy ở  Plant Pathology 2020 – FGVC7')
    st.write('Bộ dữ liệu gồm:')
    st.write('+ 1 bộ ảnh gồm n tấm ảnh')
    st.write('+ 1 file train.csv')
    st.write('+ 1 file test.csv')
    st.subheader("train.csv")
    st.write(training_data.head())
    st.subheader("test.csv")
    st.write(testing_data.head())
    st.subheader("Một vài hình ảnh trong dataset")
    st.image(healthy, caption="Healthy leaves")
    st.image(multiple_diseases, caption="Leaves with multiple diseases")
    st.image(rust, caption="Leaves with rust")
    st.image(scrab, caption="Leaves with scab")


with features:
    st.header('The features I created')
    st.write('I found dataset on...')

with model_training:
    st.header('Time to train model')
    st.write('I found dataset on...')

with test:
    st.header('Thử nhận biết một số lá')
    uploaded_file = st.file_uploader("Chọn một lá bất kỳ")
    if uploaded_file is not None:
        test_image = "dataset/images/" + uploaded_file.name
        st.image(test_image, caption="test")
    
    result = st.button('Dự đoán')

    if result:
        if test_image is None:
            st.text("Please upload an image file")
        else:
            image = Image.open(test_image)
            prediction = import_and_predict(image, model)
            if np.argmax(prediction) == 0:
                st.write("Healthy!")
            elif np.argmax(prediction) == 1:
                st.write("Multiple Diseases!")
            elif np.argmax(prediction) == 2:
                st.write("Rust!")
            else: 
                st.write("Scab!")
            st.text(prediction)