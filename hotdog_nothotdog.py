import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps

model = load_model("my_model.hdf5")

import streamlit as st

# https://towardsdatascience.com/deploying-an-image-classification-web-app-with-python-3753c46bb79
# test

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

import cv2

import numpy as np

from PIL import Image


def import_and_predict(image_data, model):
    size = (256, 256)
    image = ImageOps.fit(image_data, size)
    image = np.asarray(image)
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(image, dsize=(256, 256))

    img_reshape = img_resize[np.newaxis, ...]

    prediction = model.predict(img_reshape)

    return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)

    if prediction[0][0] > prediction[0][1]:
        st.write("It is a hotdog! Confidence rate is ", prediction[0][0] * 100, "%")
    else:
        st.write("It is not a hotdog! Confidence rate is ", prediction[0][1] * 100, "%")
