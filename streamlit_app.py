
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score, accuracy_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import base64

# Function to encode the image file to base64
def get_base64_of_image(image_file):
    with open(image_file, 'rb') as img:
        return base64.b64encode(img.read()).decode()

# Path to your image file
image_path = "bg2.jpg"

# Convert the image to a Base64 string
img_base64 = get_base64_of_image(image_path)

# Create the CSS with the Base64 encoded image
st.markdown(f"""
    <style>
    @keyframes fadeIn {{
        0% {{ opacity: 0; }}
        100% {{ opacity: 1; }}
    }}
    .main {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        color: #333;
        animation: fadeIn 2s ease-in-out;
    }}
    .title {{
        font-size: 36px;
        color: #ffffff;
        text-align: center;
        padding: 20px;
        background: linear-gradient(to right, #673AB7, #9C27B0);
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        animation: fadeIn 1.5s ease-in-out;
    }}
    .subheader {{
        font-size: 24px;
        color: #ffffff;
        margin-top: 20px;
        margin-bottom: 10px;
        background: linear-gradient(to right, #2196F3, #03A9F4);
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        animation: fadeIn 1.5s ease-in-out;
    }}
    .metric {{
        font-size: 18px;
        font-weight: bold;
        color: #ffffff;
        background: linear-gradient(to right, #FF5722, #FF7043);
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        text-align: center;
        animation: fadeIn 2s ease-in-out;
    }}
    .nav {{
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
        animation: fadeIn 1.5s ease-in-out;
    }}
    .nav input[type="radio"] {{
        display: none;
    }}
    .nav label {{
        background: linear-gradient(to right, #4CAF50, #8BC34A);
        color: #ffffff;
        padding: 12px 24px;
        border-radius: 8px;
        cursor: pointer;
        margin: 0 5px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        font-weight: bold;
        transition: background 0.3s ease, transform 0.3s ease;
    }}
    .nav input[type="radio"]:checked + label {{
        background: linear-gradient(to right, #333333, #616161);
        transform: scale(1.05);
    }}
    .nav label:hover {{
        transform: scale(1.05);
    }}
    .content {{
        font-size: 22px;
        line-height: 1.8;
        color: #333;
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        animation: fadeIn 2s ease-in-out;
    }}
    .content h2 {{
        font-size: 32px;
        color: #673AB7;
        text-align: center;
    }}
    .content h3 {{
        font-size: 26px;
        color: #2196F3;
    }}
    .content ul {{
        font-size: 20px;
        margin: 10px 0;
        padding: 0;
    }}
    .content ul li {{
        margin-bottom: 10px;
    }}
    .content a {{
        color: #673AB7;
        text-decoration: none;
        font-weight: bold;
    }}
    .content a:hover {{
        text-decoration: underline;
    }}
    .form-container {{
        font-size: 22px;
        line-height: 1.8;
        color: #333;
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        animation: fadeIn 2s ease-in-out;
    }}
    .form-container form {{
        display: flex;
        flex-direction: column;
    }}
    .form-container label {{
        font-size: 20px;
        margin-top: 10px;
    }}
    .form-container input, .form-container textarea {{
        padding: 10px;
        margin-top: 5px;
        border: 1px solid #ddd;
        border-radius: 8px;
        font-size: 18px;
    }}
    .form-container input[type="submit"] {{
        background: linear-gradient(to right, #673AB7, #9C27B0);
        color: #ffffff;
        padding: 12px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 20px;
        font-weight: bold;
        transition: background 0.3s ease, transform 0.3s ease;
    }}
    .form-container input[type="submit"]:hover {{
        background: linear-gradient(to right, #333333, #616161);
        transform: scale(1.05);
    }}
    </style>
""", unsafe_allow_html=True)

# Page navigation
page = st.radio("", ["About Us", "Home", "Contact Us"], index=1, horizontal=True, key='nav')
if page == "About Us":
    st.markdown('<div class="title">About Us</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='font-size: 22px; line-height: 1.6; color: #333; background-color: rgba(255, 255, 255, 0.9); padding: 15px; border-radius: 8px;'>
    <h2 style='font-size: 32px; color: #673AB7;'>Welcome to General Index Forecasting!</h2>

    We are a dedicated team of data scientists and analysts passionate about harnessing the power of data to drive informed decisions. Our mission is to provide actionable insights through advanced forecasting techniques and data-driven analysis.

    <h3 style='font-size: 26px; color: #2196F3;'>Our Expertise:</h3>
    <ul style='font-size: 20px;'>
    <li><strong>Data Analytics:</strong> Transforming raw data into meaningful insights.</li>
    <li><strong>Machine Learning:</strong> Developing predictive models to anticipate future trends.</li>
    <li><strong>Business Intelligence:</strong> Leveraging data to enhance business strategies.</li>
    </ul>

    <h3 style='font-size: 26px; color: #2196F3;'>Our Collaboration with MOSPI:</h3>
    This project is specifically designed to cater to the needs of the Ministry of Statistics and Programme Implementation (MOSPI). We aim to provide MOSPI with advanced forecasting tools and insights to support their data-driven decision-making processes.

    <h3 style='font-size: 26px; color: #2196F3;'>Our Team:</h3>
    <ul style='font-size: 20px;'>
    <li><strong>Sai Manasa B</strong> - Data Scientist: With over 4 years of experience in data science, Sai specializes in developing machine learning models and data analysis.</li>
    <li><strong>Kimberly Marclin Nathan A</strong> - Data Analyst: Kim has a knack for turning complex data into clear and actionable insights, integrating data analytics into business strategies to drive growth and efficiency.</li>
    </ul>
