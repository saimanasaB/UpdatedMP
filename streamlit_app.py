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
image_path = "bg.jpg"

# Convert the image to a Base64 string
img_base64 = get_base64_of_image(image_path)
st.markdown(f"""
    <style>
    .main {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        color: #f4f4f4;
    }}
    .title {{
        font-size: 48px;
        color: #FFFFFF;
        text-align: center;
        padding: 20px;
        background: linear-gradient(to right, #FF5722, #FF9800);
        border-radius: 10px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.5);
        animation: fadeInDown 1s ease-out;
        font-family: 'Montserrat', sans-serif;
    }}
    .subheader {{
        font-size: 32px;
        color: #FFFFFF;
        margin-top: 20px;
        margin-bottom: 10px;
        background: linear-gradient(to right, #4CAF50, #8BC34A);
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        animation: fadeInUp 1s ease-out;
        font-family: 'Roboto', sans-serif;
    }}
    .metric {{
        font-size: 22px;
        font-weight: bold;
        color: #FFFFFF;
        background: linear-gradient(to right, #673AB7, #9C27B0);
        padding: 12px;
        border-radius: 10px;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);
        text-align: center;
        transition: transform 0.3s ease, background 0.3s ease;
        font-family: 'Open Sans', sans-serif;
    }}
    .metric:hover {{
        transform: translateY(-7px);
        background: linear-gradient(to right, #7E57C2, #B39DDB);
    }}
    .nav {{
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
        animation: fadeIn 1.5s ease-out;
    }}
    .nav input[type="radio"] {{
        display: none;
    }}
    .nav label {{
        background: linear-gradient(to right, #3F51B5, #5C6BC0);
        color: #FFFFFF;
        padding: 15px 30px;
        border-radius: 8px;
        cursor: pointer;
        margin: 0 5px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        font-weight: bold;
        transition: background 0.3s ease, transform 0.3s ease;
        font-family: 'Lato', sans-serif;
    }}
    .nav label:hover {{
        background: linear-gradient(to right, #283593, #3949AB);
        transform: scale(1.05);
    }}
    .nav input[type="radio"]:checked + label {{
        background: linear-gradient(to right, #1C1C1C, #616161);
    }}
    .content {{
        font-size: 26px;
        line-height: 1.8;
        color: #333;
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        animation: fadeIn 2s ease-out;
        font-family: 'Poppins', sans-serif;
    }}
    .content h2 {{
        font-size: 38px;
        color: #FF9800;
        text-align: center;
        animation: fadeInDown 1s ease-out;
    }}
    .content h3 {{
        font-size: 32px;
        color: #4CAF50;
        animation: fadeInUp 1s ease-out;
    }}
    .content ul {{
        font-size: 24px;
        margin: 10px 0;
        padding: 0;
        animation: fadeIn 2s ease-out;
    }}
    .content ul li {{
        margin-bottom: 15px;
    }}
    .content a {{
        color: #673AB7;
        text-decoration: none;
        font-weight: bold;
        transition: color 0.3s ease, transform 0.3s ease;
    }}
    .content a:hover {{
        color: #311B92;
        transform: scale(1.05);
    }}
    .form-container {{
        font-size: 26px;
        line-height: 1.8;
        color: #333;
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        animation: fadeIn 2s ease-out;
        font-family: 'Muli', sans-serif;
    }}
    .form-container form {{
        display: flex;
        flex-direction: column;
    }}
    .form-container label {{
        font-size: 24px;
        margin-top: 10px;
    }}
    .form-container input, .form-container textarea {{
        padding: 12px;
        margin-top: 5px;
        border: 1px solid #ddd;
        border-radius: 8px;
        font-size: 22px;
        transition: border 0.3s ease, box-shadow 0.3s ease;
    }}
    .form-container input:focus, .form-container textarea:focus {{
        border-color: #673AB7;
        box-shadow: 0 0 8px rgba(103, 58, 183, 0.5);
    }}
    .form-container input[type="submit"] {{
        background: linear-gradient(to right, #673AB7, #9C27B0);
        color: #ffffff;
        padding: 15px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 24px;
        font-weight: bold;
        transition: background 0.3s ease, transform 0.3s ease;
    }}
    .form-container input[type="submit"]:hover {{
        background: linear-gradient(to right, #5E35B1, #7B1FA2);
        transform: translateY(-5px);
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    @keyframes fadeInDown {{
        from {{ opacity: 0; transform: translateY(-20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
""", unsafe_allow_html=True)

# Page navigation
page = st.radio("", ["About Us", "Home", "Contact Us"], index=1, horizontal=True, key='nav')

if page == "About Us":
    st.markdown('<div class="title">About Us</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="content">
    <h2>Welcome to General Index Forecasting!</h2>
    We are a dedicated team of data scientists and analysts passionate about harnessing the power of data to drive informed decisions. Our mission is to provide actionable insights through advanced forecasting techniques and data-driven analysis.
    
    <h3>Our Expertise:</h3>
    <ul>
    <li><strong>Data Analytics:</strong> Transforming raw data into meaningful insights.</li>
    <li><strong>Machine Learning:</strong> Developing predictive models to anticipate future trends.</li>
    <li><strong>Business Intelligence:</strong> Leveraging data to enhance business strategies.</li>
    </ul>
    
    <h3>Our Collaboration with MOSPI:</h3>
    This project is specifically designed to cater to the needs of the Ministry of Statistics and Programme Implementation (MOSPI). We aim to provide MOSPI with advanced forecasting tools and insights to support their data-driven decision-making process.
    
    <p>For more information about our work, please visit our <a href="https://example.com" target="_blank">website</a>.</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "Home":
    st.markdown('<div class="title">Home</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="content">
    <h2>Welcome to Our Forecasting App</h2>
    Our app offers advanced forecasting capabilities using both SARIMA and LSTM models. Explore our features and visualize forecasts with ease.
    
    <h3>Key Features:</h3>
    <ul>
    <li><strong>SARIMA Forecasting:</strong> Seasonal AutoRegressive Integrated Moving Average model for time series analysis.</li>
    <li><strong>LSTM Forecasting:</strong> Long Short-Term Memory model for predicting future values.</li>
    </ul>
    
    <p>Use the navigation menu to explore different sections of the app.</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "Contact Us":
    st.markdown('<div class="title">Contact Us</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="form-container">
    <h2>Get in Touch</h2>
    <p>We'd love to hear from you! Please fill out the form below, and we'll get back to you as soon as possible.</p>
    
    <form action="mailto:your-email@example.com" method="post" enctype="text/plain">
        <label for="name">Name:</label>
        <input type="text" id="name" name="name" required>
        
        <label for="email">Email:</label>
        <input type="email" id="email" name="email" required>
        
        <label for="message">Message:</label>
        <textarea id="message" name="message" rows="6" required></textarea>
        
        <input type="submit" value="Send">
    </form>
    </div>
    """, unsafe_allow_html=True)
