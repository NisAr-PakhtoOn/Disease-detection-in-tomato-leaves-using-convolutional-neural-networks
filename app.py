import streamlit as st
from PIL import Image, ImageOps
import numpy as np
# from io import BytesIO
from PIL import Image
from keras.models import load_model



st.set_page_config(page_title='Machine Learning App with Random Forest')




st.title("Disease Detection in Tomato leaves")
st.text("Upload an image of tomato leaf")
# from img_classification import teachable_machine_classification
MODEL = load_model("Tomato.h5")
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
st.info(predicted_class)
st.text('Confidence')
st.info(confidence)

st.text('Medicine for a quick treatment')

if predicted_class == 'Tomato_Bacterial_spot':
    st.info('A plant with bacterial spot cannot be cured. Remove symptomatic plants from the field or greenhouse to prevent the spread of bacteria to healthy plants. Burn, bury or hot compost the affected plants and DO NOT eat symptomatic fruit.')
elif predicted_class == 'Tomato_Early_blight':
    st.info('Cure the plant quickly otherwise the diease can be spread, Thoroughly spray the plant (bottoms of leaves also) with Bonide Liquid Copper Fungicide concentrate or Bonide Tomato & Vegetable')
elif predicted_class == 'Tomato_Late_blight':
    st.info('Spraying fungicides is the most effective way to prevent late blight. For conventional gardeners and commercial producers, protectant fungicides such as chlorothalonil (e.g., Bravo, Echo, Equus, or Daconil) and Mancozeb (Manzate) can be used.')
elif predicted_class == 'Tomato_Leaf_Mold':
    st.info('Baking soda solution: Mix 1 tablespoon baking soda and ½ teaspoon liquid soap such as Castile soap (not detergent) in 1 gallon of water. Spray liberally, getting top and bottom leaf surfaces and any affected areas.')
elif predicted_class == 'Tomato_Septoria_leaf_spot':
    st.info('fungicides with active ingredients such as chlorothalonil, copper, or mancozeb will help reduce disease, but they must be applied before disease occurs as they can only provide preventative protection. They will not cure the plant. If the disease has spread than remove the plants')
elif predicted_class == 'Tomato_Spider_mites_Two_spotted_spider_mite':
    st.info('aiming a hard stream of water at infested plants to knock spider mites off the plants. Other options include insecticidal soaps, horticultural oils, or neem oil.')
elif predicted_class == 'Tomato__Target_Spot':
    st.info('Products containing chlorothalonil, mancozeb, and copper oxychloride have been shown to provide good control of target spot in research trials.')
elif predicted_class == 'Tomato__Tomato_YellowLeaf__Curl_Virus':
    st.info('Use a neonicotinoid insecticide, such as dinotefuran (Venom) imidacloprid (AdmirePro, Alias, Nuprid, Widow, and others) or thiamethoxam (Platinum), as a soil application or through the drip irrigation system at transplanting of tomatoes or peppers.')
elif predicted_class == 'Tomato__Tomato_mosaic_virus':
    st.info('Remove all infected plants and destroy them. Do NOT put them in the compost pile, as the virus may persist in infected plant matter. Monitor the rest of your plants closely, especially those that were located near infected plants. Disinfect gardening tools after every use.')
elif predicted_class == 'Tomato_healthy':
    st.info('Your plant is healthy, there is no need to apply medicines, please take care of your plants, if any disease occurs, than cure it fast and remove the infected leaves.')




