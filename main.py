import streamlit as st
import requests
from keras.models import load_model
from PIL import Image
import numpy as np

from util import set_background


set_background('./bgs/bg.png')

# set title
st.markdown('<h1 style="color: white;">Disease Detection in Orange Leaves</h1>', unsafe_allow_html=True)

# set header
st.markdown('<h2 style="color: white;">Please upload image</h2>', unsafe_allow_html=True)

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

def predict(image):
    # Send a request to the Cloud Function URL with the image data
    cloud_function_url = "https://us-central1-orange-leaves.cloudfunctions.net/predict"
    response = requests.post(cloud_function_url, files={"file": image})
    
    # Parse the response JSON
    predictions = response.json()

    print("Predictions:", predictions)  # Debugging print statement
    
    return predictions

# display image
if file is not None:
    image = file.read()
    st.image(image, use_column_width=True)

    # write classification
    predictions = predict(image)
    predicted_class = predictions.get('class', 'Unknown')
    confidence = predictions.get('confidence', 0.0)
    st.write('<h2 style="color: white;">Predicted Disease: {}</h2>'.format(predicted_class), unsafe_allow_html=True)
    st.write('<h3 style="color: white;">Confidence score: {}%</h3>'.format(int(confidence)), unsafe_allow_html=True)
    
