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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# Import LSTM packages
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Import SARIMA packages
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



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
    <div style='font-size: 22px; line-height: 1.6; color: #333; background-color: rgba(255, 255, 255, 0.9); padding: 15px; border-radius: 8px;'>
    <h2 style='font-size: 32px; color: #4CAF50;'>Welcome to General Index Forecasting!</h2>

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

    <h3 style='font-size: 26px; color: #2196F3;'>Our Vision:</h3>
    We aim to be at the forefront of data science and analytics, continuously innovating to provide our clients, including MOSPI, with the best tools and strategies for success. Our goal is to empower organizations with the knowledge and foresight to make data-driven decisions.

    <h3 style='font-size: 26px; color: #2196F3;'>Our Values:</h3>
    <ul style='font-size: 20px;'>
    <li><strong>Integrity:</strong> We uphold the highest standards of honesty and transparency.</li>
    <li><strong>Innovation:</strong> We embrace creativity and new ideas to solve complex problems.</li>
    <li><strong>Excellence:</strong> We are committed to delivering high-quality results and solutions.</li>
    </ul>

    <h3 style='font-size: 26px; color: #2196F3;'>Get Involved:</h3>
    We are always looking to collaborate with like-minded professionals and organizations. If you're interested in working with us or learning more about our services, please reach out through our contact page.

    <br><br><center>Thank you for visiting our website and learning more about us!</center>
    </div>
    """, unsafe_allow_html=True)

elif page == "Home":
    st.markdown('<div class="title">General Index Forecasting using LSTM and SARIMA</div>', unsafe_allow_html=True)
    
    # Load the dataset
    file_path = st.text_input('Enter file path of cleaned data (e.g., cleaned_data.csv)', 'cleaned_data.csv')
    data = pd.read_csv(file_path)
    # Scaling the data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

# Train-Test Split
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

def create_dataset(dataset, time_step=12):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

# Create LSTM data
time_step = 12
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# LSTM Model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform to get original values
train_predict = scaler.inverse_transform(train_predict)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate LSTM Model
mse_lstm = mean_squared_error(y_test_inv, test_predict)
mae_lstm = mean_absolute_error(y_test_inv, test_predict)
rmse_lstm = math.sqrt(mse_lstm)
print(f"LSTM Test RMSE: {rmse_lstm}, MAE: {mae_lstm}")

# Plot LSTM predictions
plt.figure(figsize=(10,6))
plt.plot(y_test_inv, label='True')
plt.plot(test_predict, label='LSTM Predictions')
plt.title('LSTM Predictions vs True Data')
plt.legend()
plt.show()

# SARIMA Model
# Plot ACF and PACF to help estimate the order
plot_acf(data, lags=40)
plot_pacf(data, lags=40)
plt.show()

# Choosing parameters based on ACF/PACF (Adjust p, d, q, P, D, Q based on your data)
sarima_model = SARIMAX(data[:train_size], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit(disp=False)

# Forecasting
sarima_forecast = sarima_result.forecast(steps=len(test_data))

# Evaluate SARIMA Model
mse_sarima = mean_squared_error(data[train_size:].values, sarima_forecast)
mae_sarima = mean_absolute_error(data[train_size:].values, sarima_forecast)
rmse_sarima = math.sqrt(mse_sarima)
print(f"SARIMA Test RMSE: {rmse_sarima}, MAE: {mae_sarima}")

# Plot SARIMA forecast
plt.figure(figsize=(10,6))
plt.plot(data[train_size:], label='True')
plt.plot(sarima_forecast, label='SARIMA Predictions')
plt.title('SARIMA Predictions vs True Data')
plt.legend()
plt.show()

# Comparison Plot
plt.figure(figsize=(10,6))
plt.plot(y_test_inv, label='True')
plt.plot(test_predict, label='LSTM Predictions')
plt.plot(sarima_forecast, label='SARIMA Predictions')
plt.title('LSTM vs SARIMA Predictions')
plt.legend()
plt.show()



elif page == "Contact Us":
    st.markdown('<div class="title">Contact Us</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='font-size: 22px; line-height: 1.6; color: #333; background-color: rgba(255, 255, 255, 0.9); padding: 15px; border-radius: 8px;'>
    <h2 style='font-size: 32px; color: #4CAF50;'>We'd Love to Hear from You!</h2>

    Whether you have questions, feedback, or are interested in our services, please don't hesitate to reach out to us. We're here to help!

    <h3 style='font-size: 26px; color: #2196F3;'>Contact Information:</h3>
    <ul style='font-size: 20px;'>
    <li><strong>Email:</strong> <a href="mailto:kimsai@gmail.com" style="color: #4CAF50;">kimsai@gmail.com</a></li>
    <li><strong>Phone:</strong> +91-948-678-890</li>
    <li><strong>Address:</strong> 123 Data Street, Analytics City, DataLand</li>
    </ul>

    <h3 style='font-size: 26px; color: #2196F3;'>Business Hours:</h3>
    <ul style='font-size: 20px;'>
    <li><strong>Monday to Friday:</strong> 9:00 AM - 6:00 PM (IST)</li>
    <li><strong>Saturday:</strong> 10:00 AM - 4:00 PM (IST)</li>
    <li><strong>Sunday:</strong> Closed</li>
    </ul>

    <h3 style='font-size: 26px; color: #2196F3;'>Follow Us:</h3>
    <ul style='font-size: 20px;'>
    <li><a href="https://www.linkedin.com/in/sai-manasa-b-1765b420b/" style="color: #4CAF50;">LinkedIn</a></li>
    <li><a href="https://twitter.com/example" style="color: #4CAF50;">Twitter</a></li>
    <li><a href="https://facebook.com/example" style="color: #4CAF50;">Facebook</a></li>
    </ul>
    <h3 style='font-size: 26px; color: #2196F3;'>Get In Touch:</h3>
    If you have any inquiries or would like to discuss potential projects, please fill out the contact form below or use the contact details provided.<br><br>
    <h3 style='font-size: 22px; color: #2196F3;'>Feedback Form:</h3>
    <p>We appreciate your feedback. Please fill out the form below:</p>
    
    <form action="https://example.com/feedback" method="post">
        <label for="name">Name:</label><br>
        <input type="text" id="name" name="name" required><br>
        <label for="email">Email:</label><br>
        <input type="email" id="email" name="email" required><br>
        <label for="message">Message:</label><br>
        <textarea id="message" name="message" rows="4" required></textarea><br>
        <input type="submit" value="Submit">
    </form>

    Thank you for your interest in connecting with us!
    </div>
    """, unsafe_allow_html=True)
