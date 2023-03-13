# data ref: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset?resource=download


import streamlit as st
from streamlit_lottie import st_lottie
import pickle
import requests


st.set_page_config(layout='wide')


# Customizing lottie logo dimensions

st.sidebar.markdown("Customize lottie logo dimensions")
lottie_height = st.sidebar.slider(
    "Lottie height", 100, 1000, 250)
lottie_width = st.sidebar.slider(
    "Lottie width", 100, 1000, 550)
lottie_speed = st.sidebar.slider(
    "Lottie speed", 1, 10, 5)


# Adding lottie logo to app

url = requests.get(
    'https://assets10.lottiefiles.com/packages/lf20_Lpuvp7YT5K.json')
url_json = dict()
if url.status_code == 200:
    url_json = url.json()
else:
    print("Error in URL")

st_lottie(
    url_json,
    height=lottie_height,
    speed=lottie_speed,
    width=lottie_width)


# App title, etc

st.title("AI Value Chain Process Optimizer (AI-VPO)")
st.header("* Crop Recommender (MVP Demo)")
st.subheader("Description:")
st.subheader("This is your personal AI-assistant, "
             "specialized for crop recommendation "
             "on the basis of the current state of "
             "your farm.")
st.subheader(" ")


# Load model and outputs

rf_pickle = open('ocz.pickle', 'rb')
map_pickle = open('output_ocz.pickle', 'rb')

rfc = pickle.load(rf_pickle)
unique_crop_mapping = pickle.load(map_pickle)

# st.write(rfc)
# st.write(unique_crop_mapping)

rf_pickle.close()
map_pickle.close()


# Define user inputs for the app

nitrogen = st.number_input(
    'Please input the ratio of soil Nitrogen, e.g. 94, 69, 78',
    min_value=0)

phosphorus = st.number_input(
    'Please input the ratio of soil Phosphorus, e.g. 42, 58, 35',
    min_value=0)

potassium = st.number_input(
    'Please input the ratio of soil Potassium, e.g. 43, 41, 36',
    min_value=0)

temperature = st.number_input(
    'Please input the environmental temperature (in degrees Celsius), e.g. 20.87, 22.70',
    min_value=0)

humidity = st.number_input(
    'Please input the environmental relative humidity (in %), e.g. 82.00, 23.37',
    min_value=0)

soil_ph = st.number_input(
    'Please input the soil pH, e.g. 6.50, 4.00',
    min_value=0)

rainfall = st.number_input(
    'Please input the average rainfall value (in mm), e.g. 202.93, 271.32',
    min_value=0)


st.write('The user inputs are {}'.format(
    [
        nitrogen, phosphorus, potassium,
        temperature, humidity, soil_ph,
        rainfall]))


# Predict optimal crop

new_prediction = rfc.predict([[
        nitrogen, phosphorus, potassium,
        temperature, humidity, soil_ph,
        rainfall]])

prediction_crop = unique_crop_mapping[new_prediction][0]

st.write('With respect to the state of your farm, the most suitable crop is: <<  {}  >>'.format(
    prediction_crop))
