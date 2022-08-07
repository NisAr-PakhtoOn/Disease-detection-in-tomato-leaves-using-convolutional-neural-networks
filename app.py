import streamlit as st
from PIL import Image, ImageOps
import numpy as np
# from io import BytesIO
from PIL import Image
import tensorflow as tf

st.title("Disease Detection in Tomato leaves")
st.header("Disease Detection in Tomato leaves")
st.text("Upload an image of tomato leaf")
# from img_classification import teachable_machine_classification
MODEL = tf.keras.models.load_model("Tomato.h5")
CLASS_NAMES = ['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']
uploaded_file = st.file_uploader("Choose an image ...", type="jpg")

image = Image.open(uploaded_file)

st.image(image, caption='Tomato Leaf Image')
image = image.resize((256,256))
img_batch = np.expand_dims(image, 0)
# # img_batch = np.asarray(uploaded_file, 0).astype('float32')

predictions = MODEL.predict(img_batch)
predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
confidence = np.max(predictions[0])
st.text('Prediction')
st.text(predicted_class)
st.text('Confidence')
st.text(confidence)

st.text('Medicine for a quick treatment')

if predicted_class == 'Tomato_Bacterial_spot':
    st.text('Bacterial Spot spray')
elif predicted_class == 'Tomato_Early_blight':
    st.text('Early blight spray')
elif predicted_class == 'Tomato_Late_blight':
    st.text('Late blight spray')
elif predicted_class == 'Tomato_Leaf_Mold':
    st.text('Leaf Mold spray')
elif predicted_class == 'Tomato_Septoria_leaf_spot':
    st.text('Septoria leaf spray')
elif predicted_class == 'Tomato_Spider_mites_Two_spotted_spider_mite':
    st.text('Spotted spider mite spray')
elif predicted_class == 'Tomato__Target_Spot':
    st.text('Target spot spray')
elif predicted_class == 'Tomato__Tomato_YellowLeaf__Curl_Virus':
    st.text('Curl Virus spray')
elif predicted_class == 'Tomato__Tomato_mosaic_virus':
    st.text('Mosaic Virus spray')
elif predicted_class == 'Tomato_healthy':
    st.text('Leaf is healthy')




