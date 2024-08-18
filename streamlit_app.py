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
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap');

    body {{
        font-family: 'Montserrat', sans-serif;
    }}

    .main {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        color: #f5f5f5;
        animation: fadeInBackground 2s ease-in-out;
    }}

    @keyframes fadeInBackground {{
        from {{
            opacity: 0;
        }}
        to {{
            opacity: 1;
        }}
    }}

    .title {{
        font-size: 40px;
        color: #ffffff;
        text-align: center;
        padding: 25px;
        background: linear-gradient(to right, #ff7e5f, #feb47b);
        border-radius: 12px;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
        animation: fadeInTitle 2s ease-in-out;
    }}

    @keyframes fadeInTitle {{
        from {{
            transform: translateY(-20px);
            opacity: 0;
        }}
        to {{
            transform: translateY(0);
            opacity: 1;
        }}
    }}

    .subheader {{
        font-size: 28px;
        color: #ffffff;
        margin-top: 20px;
        margin-bottom: 15px;
        background: linear-gradient(to right, #43cea2, #185a9d);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        animation: fadeInSubheader 2s ease-in-out;
    }}

    @keyframes fadeInSubheader {{
        from {{
            transform: translateY(-15px);
            opacity: 0;
        }}
        to {{
            transform: translateY(0);
            opacity: 1;
        }}
    }}

    .metric {{
        font-size: 20px;
        font-weight: bold;
        color: #ffffff;
        background: linear-gradient(to right, #ff512f, #dd2476);
        padding: 12px;
        border-radius: 10px;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        animation: fadeInMetric 2s ease-in-out;
    }}

    @keyframes fadeInMetric {{
        from {{
            transform: translateY(10px);
            opacity: 0;
        }}
        to {{
            transform: translateY(0);
            opacity: 1;
        }}
    }}

    .nav {{
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
        animation: fadeInNav 2s ease-in-out;
    }}

    @keyframes fadeInNav {{
        from {{
            opacity: 0;
        }}
        to {{
            opacity: 1;
        }}
    }}

    .nav input[type="radio"] {{
        display: none;
    }}

    .nav label {{
        background: linear-gradient(to right, #1f4037, #99f2c8);
        color: #ffffff;
        padding: 15px 30px;
        border-radius: 10px;
        cursor: pointer;
        margin: 0 8px;
        text-align: center;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.1);
        font-weight: bold;
        transition: background 0.3s ease;
    }}

    .nav input[type="radio"]:checked + label {{
        background: linear-gradient(to right, #333333, #dd1818);
        transform: scale(1.05);
    }}

    .content {{
        font-size: 24px;
        line-height: 1.8;
        color: #333;
        background: rgba(255, 255, 255, 0.85);
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.1);
        animation: fadeInContent 2s ease-in-out;
    }}

    @keyframes fadeInContent {{
        from {{
            opacity: 0;
        }}
        to {{
            opacity: 1;
        }}
    }}

    .content h2 {{
        font-size: 34px;
        color: #ff7e5f;
        text-align: center;
    }}

    .content h3 {{
        font-size: 28px;
        color: #43cea2;
    }}

    .content ul {{
        font-size: 22px;
        margin: 15px 0;
        padding: 0;
    }}

    .content ul li {{
        margin-bottom: 12px;
    }}

    .content a {{
        color: #ff7e5f;
        text-decoration: none;
        font-weight: bold;
    }}

    .content a:hover {{
        text-decoration: underline;
    }}

    .form-container {{
        font-size: 24px;
        line-height: 1.8;
        color: #333;
        background: rgba(255, 255, 255, 0.85);
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.1);
        animation: fadeInForm 2s ease-in-out;
    }}

    @keyframes fadeInForm {{
        from {{
            opacity: 0;
        }}
        to {{
            opacity: 1;
        }}
    }}

    .form-container form {{
        display: flex;
        flex-direction: column;
    }}

    .form-container label {{
        font-size: 22px;
        margin-top: 15px;
    }}

    .form-container input, .form-container textarea {{
        padding: 12px;
        margin-top: 8px;
        border: 1px solid #ddd;
        border-radius: 10px;
        font-size: 20px;
    }}

    .form-container input[type="submit"] {{
        background: linear-gradient(to right, #1f4037, #99f2c8);
        color: #ffffff;
        padding: 15px;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        font-size: 22px;
        font-weight: bold;
        transition: background 0.3s ease, transform 0.3s ease;
    }}

    .form-container input[type="submit"]:hover {{
        background: linear-gradient(to right, #333333, #dd1818);
        transform: scale(1.05);
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
    This project is specifically designed to cater to the needs of the Ministry of Statistics and Programme Implementation (MOSPI). We aim to provide MOSPI with advanced forecasting tools and insights to support their data-driven decision-making processes.

    <h3>Our Team:</h3>
    <ul>
        <li><strong>Sai Manasa B</strong> - Data Scientist: With over 4 years of experience in data science, Sai specializes in developing machine learning models and data analysis.</li>
        <li><strong>Sri Harsha T</strong> - Lead Analyst: An expert in statistical analysis and forecasting models, Sri Harsha has been a key contributor to the project.</li>
        <li><strong>Sai Kumar S</strong> - Project Manager: Sai Kumar brings his expertise in project management and business intelligence to ensure the success of our initiatives.</li>
    </ul>

    <h3>Our Vision:</h3>
    We strive to empower organizations and governments with the tools they need to make data-driven decisions that foster growth and efficiency.

    </div>
    """, unsafe_allow_html=True)

elif page == "Home":
    st.markdown('<div class="title">Home</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Data Upload and Forecasting</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(df.head())

        # Data processing and forecasting code would go here

elif page == "Contact Us":
    st.markdown('<div class="title">Contact Us</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="form-container">
    <form action="mailto:saikumar@example.com" method="post" enctype="text/plain">
        <label for="name">Name:</label>
        <input type="text" id="name" name="name" required>

        <label for="email">Email:</label>
        <input type="email" id="email" name="email" required>

        <label for="message">Message:</label>
        <textarea id="message" name="message" rows="4" required></textarea>

        <input type="submit" value="Submit">
    </form>
    </div>
    """, unsafe_allow_html=True)
