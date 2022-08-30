import streamlit as st
import pickle
import pandas as pd
import numpy as np


st.write("""
# Car Price Prediction App
This app predicts the **Car Price**!
""")
st.write('---')


# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    company = st.sidebar.selectbox('company',('Maruti','Skoda','Honda','Hyundai','Toyota','Ford','Renault','Mahindra','Tata','Chevrolet','Fiat','Datsun','Jeep','Mercedes-Benz','Mitsubishi','Audi','Volkswagen','BMW','Nissan','Lexus','Jaguar','Land','MG','Volvo','Daewoo', 'Kia' ,'Force', 'Ambassador' ,'Ashok', 'Isuzu' ,'Opel' ,'Peugeot'))
    year = st.sidebar.slider('year', 1983.0 ,2020.0, 2013.0)
    km_driven = st.sidebar.slider('km_driven', 1.0, 194000.0, 64040.72)
    fuel = st.sidebar.selectbox('fuel',('Diesel','Petrol','LPG','CNG'))
    seller_type = st.sidebar.selectbox('seller_type',('Individual','Dealer','Trustmark Dealer'))
    transmission = st.sidebar.selectbox('transmission',('Manual','Automatic'))
    owner = st.sidebar.selectbox('owner',('First Owner','Second Owner','Third Owner','Fourth & Above Owner','Test Drive Car'))
    mileage = st.sidebar.slider('mileage',10.9, 30.46,20.68)
    engine = st.sidebar.slider('engine', 793.0, 1896.0, 1221.0)
    max_power = st.sidebar.slider('max_power',37.0, 118.35,78.17)
    seats = st.sidebar.slider('seats', 2.0,14.0,5.0)
       
    data = {'company':company,
            'year': year,
            'km_driven': km_driven,
            'fuel': fuel,
            'seller_type': seller_type,
            'transmission': transmission,
            'owner': owner,
            'mileage': mileage,
            'engine': engine,
            'max_power': max_power,
            'seats': seats}
 
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Read the csv file
data_raw=pd.read_csv('cleaned_data.csv')
data = data_raw.drop(columns=['selling_price'])
df= pd.concat([input_df,data],axis=0)

# Encoding
encode=['company','fuel','seller_type','transmission','owner']
for col in encode:
   dummy = pd.get_dummies(df[col], prefix = col)
   df = pd.concat([df,dummy],axis = 1)
   del df[col]
df=df[:1]

# loading  Random Forest
pickle_in = open('RF.pkl', 'rb') 
model = pickle.load(pickle_in)


# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')


# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of price')
st.write(prediction)
st.write('---')

