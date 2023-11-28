import base64
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model

def set_background(image_file):

    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def classify(image, model, class_names):
  image = ImageOps.fit(image, (224,224), Image.Resampling.LANCZOS)
  image_array = np.asarray(image)
  norm_image_array = image_array.astype(np.float32)/127.5 - 1

  data = np.ndarray(shape=(1,224,224,3), dtype = np.float32)
  data[0]=norm_image_array

  prediction = model.predict(data)
  index=np.argmax(prediction)
  class_names=class_names[index]
  conf_score=prediction[0][index]

  return class_names,conf_score
