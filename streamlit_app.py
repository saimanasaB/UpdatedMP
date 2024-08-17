
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
image_path = "inflation7.jpg"

# Convert the image to a Base64 string
img_base64 = get_base64_of_image(image_path)

# Create the CSS with the Base64 encoded image
st.markdown(f"""
    <style>
    .main {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        color: #333;
    }}
    .title {{
        font-size: 36px;
        color: #4CAF50;
        text-align: center;
        padding: 15px;
        background-color: rgba(255, 255, 255, 0.9); /* Semi-transparent background for readability */
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .subheader {{
        font-size: 26px;
        color: #2196F3;
        margin-top: 20px;
        margin-bottom: 15px;
        background-color: rgba(255, 255, 255, 0.9);
        padding: 8px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .metric {{
        font-size: 18px;
        font-weight: bold;
        color: #FF5722;
        background-color: rgba(255, 255, 255, 0.9);
        padding: 8px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .nav {{
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }}
    .nav input[type="radio"] {{
        display: none;
    }}
    .nav label {{
        background-color: #4CAF50;
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        cursor: pointer;
        margin: 0 5px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .nav input[type="radio"]:checked + label {{
        background-color: #333;
    }}
    .content {{
        font-size: 20px;
        line-height: 1.6;
        color: #333;
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .contact-form {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .contact-form input, .contact-form textarea {{
        width: 100%;
        padding: 10px;
        margin-bottom: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
    }}
    .contact-form input[type="submit"] {{
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 18px;
    }}
    .contact-form input[type="submit"]:hover {{
        background-color: #45a049;
    }}
    </style>
""", unsafe_allow_html=True)

# Page navigation
page = st.radio("", ["About Us", "Home", "Contact Us"], index=1, horizontal=True, key='nav')
if page == "About Us":
    st.markdown('<div class="title">About Us</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="content">
    <h2 style='font-size: 34px; color: #4CAF50;'>Welcome to General Index Forecasting!</h2>

    We are a dedicated team of data scientists and analysts passionate about harnessing the power of data to drive informed decisions. Our mission is to provide actionable insights through advanced forecasting techniques and data-driven analysis.

    <h3 style='font-size: 28px; color: #2196F3;'>Our Expertise:</h3>
    <ul>
    <li><strong>Data Analytics:</strong> Transforming raw data into meaningful insights.</li>
    <li><strong>Machine Learning:</strong> Developing predictive models to anticipate future trends.</li>
    <li><strong>Business Intelligence:</strong> Leveraging data to enhance business strategies.</li>
    </ul>

    <h3 style='font-size: 28px; color: #2196F3;'>Our Collaboration with MOSPI:</h3>
    This project is specifically designed to cater to the needs of the Ministry of Statistics and Programme Implementation (MOSPI). We aim to provide MOSPI with advanced forecasting tools and insights to support their data-driven decision-making processes.

    <h3 style='font-size: 28px; color: #2196F3;'>Our Team:</h3>
    <ul>
    <li><strong>John Doe</strong> - Lead Data Scientist: With over 10 years of experience in data science, John specializes in developing machine learning models and data analysis.</li>
    <li><strong>Jane Smith</strong> - Data Analyst: Jane has a knack for turning complex data into clear and actionable insights.</li>
    <li><strong>Alex Johnson</strong> - Business Intelligence Specialist: Alex focuses on integrating data analytics into business strategies to drive growth and efficiency.</li>
    </ul>

    <h3 style='font-size: 28px; color: #2196F3;'>Our Vision:</h3>
    We aim to be at the forefront of data science and analytics, continuously innovating to provide our clients, including MOSPI, with the best tools and strategies for success. Our goal is to empower organizations with the knowledge and foresight to make data-driven decisions.

    <h3 style='font-size: 28px; color: #2196F3;'>Our Values:</h3>
    <ul>
    <li><strong>Integrity:</strong> We uphold the highest standards of honesty and transparency.</li>
    <li><strong>Innovation:</strong> We embrace creativity and new ideas to solve complex problems.</li>
    <li><strong>Excellence:</strong> We are committed to delivering high-quality results and solutions.</li>
    </ul>

    <h3 style='font-size: 28px; color: #2196F3;'>Get Involved:</h3>
    We are always looking to collaborate with like-minded professionals and organizations. If you're interested in working with us or learning more about our services, please reach out through our contact page.

    Thank you for visiting our website and learning more about us!
    </div>
    """, unsafe_allow_html=True)

elif page == "Home":
    st.markdown('<div class="title">General Index Forecasting using LSTM and SARIMA</div>', unsafe_allow_html=True)
    
    # Load the dataset
    file_path = st.text_input('Enter file path of cleaned data (e.g., cleaned_data.csv)', 'cleaned_data.csv')
    data = pd.read_csv(file_path)
    
    # Display the DataFrame
    st.subheader('Data Preview:')
    st.dataframe(data)
    
    # Select the relevant features
    data = data[['Year', 'Month', 'General index']]
    
    # Convert Year and Month into a datetime format
    data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(DAY=1))
    
    # Sort by date
    data = data.sort_values(by='Date').reset_index(drop=True)
    
    # Drop Year and Month as they are now redundant
    data = data.drop(columns=['Year', 'Month'])
    
    # Set Date as index
    data.set_index('Date', inplace=True)
    
    # Plot the General Index to understand its trend
        st.subheader('General Index Over Time')
    base_chart = alt.Chart(data.reset_index()).mark_line().encode(
        x='Date:T',
        y='General index:Q'
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(base_chart, use_container_width=True)

    # Split the data into training and testing sets
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]
    
    # SARIMA model
    st.subheader('SARIMA Forecast')
    sarima_model = SARIMAX(train['General index'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_fit = sarima_model.fit(disp=False)
    sarima_forecast = sarima_fit.get_forecast(steps=len(test))
    sarima_forecast_index = test.index
    sarima_forecast_mean = sarima_forecast.predicted_mean
    sarima_forecast_ci = sarima_forecast.conf_int()
    
    sarima_forecast_df = pd.DataFrame({
        'Date': sarima_forecast_index,
        'Forecast': sarima_forecast_mean,
        'Lower CI': sarima_forecast_ci.iloc[:, 0],
        'Upper CI': sarima_forecast_ci.iloc[:, 1]
    }).set_index('Date')
    
    # Plot SARIMA forecast
    sarima_chart = alt.Chart(sarima_forecast_df.reset_index()).mark_line().encode(
        x='Date:T',
        y='Forecast:Q',
        color=alt.value('blue')
    ).properties(
        width=800,
        height=400
    )
    
    actual_chart = alt.Chart(test.reset_index()).mark_line().encode(
        x='Date:T',
        y='General index:Q',
        color=alt.value('red')
    )
    
    combined_chart = sarima_chart + actual_chart
    st.altair_chart(combined_chart, use_container_width=True)
    
    # LSTM model
    st.subheader('LSTM Forecast')
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train[['General index']])
    scaled_test = scaler.transform(test[['General index']])
    
    X_train, y_train = [], []
    for i in range(60, len(scaled_train)):
        X_train.append(scaled_train[i-60:i])
        y_train.append(scaled_train[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    
    X_test, y_test = [], []
    for i in range(60, len(scaled_test)):
        X_test.append(scaled_test[i-60:i])
        y_test.append(scaled_test[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)
    
    # Create DataFrame for LSTM forecast
    lstm_forecast_df = pd.DataFrame({
        'Date': test.index[60:],
        'Forecast': y_pred.flatten()
    }).set_index('Date')
    
    # Plot LSTM forecast
    lstm_chart = alt.Chart(lstm_forecast_df.reset_index()).mark_line().encode(
        x='Date:T',
        y='Forecast:Q',
        color=alt.value('green')
    ).properties(
        width=800,
        height=400
    )
    
    lstm_actual_chart = alt.Chart(test.reset_index()).mark_line().encode(
        x='Date:T',
        y='General index:Q',
        color=alt.value('red')
    )
    
    lstm_combined_chart = lstm_chart + lstm_actual_chart
    st.altair_chart(lstm_combined_chart, use_container_width=True)

elif page == "Contact Us":
    st.markdown('<div class="title">Contact Us</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="contact-form">
    <h2 style='font-size: 34px; color: #4CAF50;'>Get in Touch</h2>
    <form action="mailto:your-email@example.com" method="post" enctype="text/plain">
        <label for="name">Name:</label>
        <input type="text" id="name" name="name" required>
        <label for="email">Email:</label>
        <input type="email" id="email" name="email" required>
        <label for="message">Message:</label>
        <textarea id="message" name="message" rows="4" required></textarea>
        <input type="submit" value="Send">
    </form>
    </div>
    """, unsafe_allow_html=True)
