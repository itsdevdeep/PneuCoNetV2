import base64
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model
from utilities import set_background, classify

st.set_option('deprecation.showfileUploaderEncoding',False)

set_background('./app-background/XRay.png')

st.title("Pulmonary Disease Detection using PneuCoNet")
st.header("Provide a clear Chest X-Ray sample")

file = st.file_uploader("Upload the X-Ray", type=['jpg','jpeg','png'])
model = load_model('./model/pneuconet.hdf5')

class_names = ['COVID','Normal','Pneumonia']

if file is not None:
  image=Image.open(file).convert('RGB')
  st.image(image, caption='Uploaded Image', use_column_width=False, width=image.width)

  class_names, conf_score = classify(image, model, class_names)

  if(st.button("Predict")):
    st.write('### Predictions:')
    st.write(f"The provided Chest X-Ray is most likely a  {class_names} sample")
    st.success(f"The Confidence Score is {conf_score * 100:.2f}%")

    if class_names in ["Pneumonia", "COVID"]:
      st.warning("Immediate attention might be required. Please consult a healthcare professional.",icon="⚠️")
    else:
      st.warning("Although you appear to be healthy, consider seeking medical attention.")
