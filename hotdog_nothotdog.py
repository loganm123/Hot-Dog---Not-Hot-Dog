import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps

model = load_model("my_model.hdf5")

import streamlit as st

# https://towardsdatascience.com/deploying-an-image-classification-web-app-with-python-3753c46bb79

st.write(
    """
    #Hot Dog Classifier
"""
)

st.write(
    """
    This is a simple image classification web app that can tell you if an image is a hot dog or not
"""
)

file = st.file_uploader("Please upload an image file (jpg or png)", type=["jpg", "png"])

import cv2 as cv

import numpy as np

from PIL import Image
from keras.preprocessing.image import img_to_array


def import_and_predict(image_data, model):
    # size = (256,256)
    # image_ingest = tf.image.decode_jpeg(image_data, channels = 3)
    # image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    # img = Image.open(image_data).resize(256,256)
    # img = np.array(img)
    # image = tf.image.resize(image_ingest,[256,256])
    # prediction = model.predict(image, batch_size = 1)
    #img = img_to_array(image_data)
    #img = tf.keras.utils.load_img(image_data, target_size=(256, 256), color_mode="rgb")
    #img = tf.image.resize(image_data, (256,256))
    #img_tensor = tf.keras.utils.img_to_array(img)
    #img_tensor = keras.utils.normalize(img_tensor, axis=1)
    prediction = model.predict(tf.expand_dims(image_tensor, axis=0))
    return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)

    if prediction[0][0] > prediction[0][1]:
        st.write("It is a hotdog!")
    else:
        st.write("It is not a hotdog!")
    st.write(prediction)
