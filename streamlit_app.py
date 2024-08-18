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

# Create the CSS with the Base64 encoded image and enhanced styling
st.markdown(f"""
    <style>
    .main {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        color: #333;
        animation: fadeIn 2s ease-in-out;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}

    .title {{
        font-size: 36px;
        color: #ffffff;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, rgba(0, 176, 255, 1) 0%, rgba(0, 204, 255, 1) 50%, rgba(0, 230, 255, 1) 100%);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        animation: slideIn 1s ease-out;
    }}

    @keyframes slideIn {{
        from {{ transform: translateX(-100%); }}
        to {{ transform: translateX(0); }}
    }}

    .subheader {{
        font-size: 24px;
        color: #ffffff;
        margin-top: 20px;
        margin-bottom: 10px;
        background: linear-gradient(90deg, rgba(255, 64, 129, 1) 0%, rgba(255, 128, 171, 1) 50%, rgba(255, 182, 193, 1) 100%);
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        animation: fadeIn 2s ease-in-out;
    }}

    .metric {{
        font-size: 18px;
        font-weight: bold;
        color: #ffffff;
        background: linear-gradient(90deg, rgba(255, 152, 0, 1) 0%, rgba(255, 193, 7, 1) 50%, rgba(255, 235, 59, 1) 100%);
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        animation: bounceIn 1.5s ease-in-out;
    }}

    @keyframes bounceIn {{
        0%, 20%, 50%, 80%, 100% {{
            transform: translateY(0);
        }}
        40% {{
            transform: translateY(-30px);
        }}
        60% {{
            transform: translateY(-15px);
        }}
    }}

    .nav {{
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
        animation: slideIn 1s ease-out;
    }}

    .nav input[type="radio"] {{
        display: none;
    }}

    .nav label {{
        background: linear-gradient(90deg, rgba(33, 150, 243, 1) 0%, rgba(41, 182, 246, 1) 50%, rgba(66, 220, 252, 1) 100%);
        color: #ffffff;
        padding: 12px 24px;
        border-radius: 8px;
        cursor: pointer;
        margin: 0 5px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        font-weight: bold;
        transition: background 0.3s ease, transform 0.3s ease;
    }}

    .nav label:hover {{
        transform: scale(1.05);
    }}

    .nav input[type="radio"]:checked + label {{
        background: linear-gradient(90deg, rgba(76, 175, 80, 1) 0%, rgba(129, 199, 132, 1) 50%, rgba(173, 255, 47, 1) 100%);
    }}

    .content {{
        font-size: 22px;
        line-height: 1.8;
        color: #333;
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        animation: fadeIn 2s ease-in-out;
    }}

    .content h2 {{
        font-size: 32px;
        color: #4CAF50;
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
        color: #4CAF50;
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
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
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
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }}

    .form-container input:focus, .form-container textarea:focus {{
        border-color: #4CAF50;
        box-shadow: 0 0 8px rgba(76, 175, 80, 0.5);
    }}

    .form-container input[type="submit"] {{
        background: linear-gradient(90deg, rgba(0, 176, 255, 1) 0%, rgba(0, 230, 255, 1) 100%);
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
        background: linear-gradient(90deg, rgba(33, 150, 243, 1) 0%, rgba(66, 220, 252, 1) 100%);
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
    <h2 style='font-size: 32px; color: #4CAF50;'>Welcome to General Index Forecasting!</h2>

    We are a dedicated team of data scientists and analysts passionate about harnessing the power of data to drive informed decisions. Our mission is to provide actionable insights through advanced forecasting techniques and data-driven analysis.

    <h3 style='font-size: 26px; color: #2196F3;'>Our Services</h3>
    <ul>
        <li>Advanced Time Series Forecasting</li>
        <li>Predictive Modeling with Machine Learning</li>
        <li>Comprehensive Data Visualization</li>
    </ul>

    <h3 style='font-size: 26px; color: #2196F3;'>Our Approach</h3>
    We combine state-of-the-art machine learning algorithms with domain expertise to deliver accurate and reliable forecasts. Our models are built and validated using industry-standard techniques to ensure the highest quality of predictions.

    <h3 style='font-size: 26px; color: #2196F3;'>Why Choose Us?</h3>
    <ul>
        <li>Expertise in both traditional statistical methods and modern machine learning techniques.</li>
        <li>Commitment to delivering data-driven insights tailored to your business needs.</li>
        <li>Customized solutions and a client-centric approach to problem-solving.</li>
    </ul>

    <p>Explore our services and discover how we can help you make informed decisions through data-driven forecasting.</p>
    <p>Contact us today to learn more about our offerings and how we can assist you.</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "Home":
    st.markdown('<div class="title">General Index Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Performance Metrics</div>', unsafe_allow_html=True)

    # Example Data (Replace with actual data)
    np.random.seed(0)
    dates = pd.date_range(start='2020-01-01', periods=100, freq='M')
    data = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.5, 100)
    df = pd.DataFrame({'Date': dates, 'Value': data})
    df.set_index('Date', inplace=True)

    # Preprocess Data for LSTM and SARIMA
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['Scaled_Value'] = scaler.fit_transform(df[['Value']])

    # Train/Test Split
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    # SARIMA Model
    sarima_model = SARIMAX(train['Value'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_result = sarima_model.fit(disp=False)
    sarima_forecast = sarima_result.predict(start=test.index[0], end=test.index[-1])
    
    # LSTM Model
    def create_lstm_model(input_shape):
        model = Sequential([
            LSTM(50, activation='relu', input_shape=input_shape),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def prepare_data_for_lstm(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 10
    X_train, y_train = prepare_data_for_lstm(train[['Scaled_Value']].values, time_step)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    lstm_model = create_lstm_model((X_train.shape[1], 1))
    lstm_model.fit(X_train, y_train, epochs=50, verbose=0)

    # Forecast with LSTM
    X_test, y_test = prepare_data_for_lstm(df[['Scaled_Value']].values, time_step)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    lstm_forecast = lstm_model.predict(X_test)
    lstm_forecast = scaler.inverse_transform(lstm_forecast)

    # Metrics
    mse = mean_squared_error(test['Value'], sarima_forecast)
    mae = mean_absolute_error(test['Value'], sarima_forecast)
    precision = precision_score(test['Value'] > test['Value'].median(), sarima_forecast > test['Value'].median())
    recall = recall_score(test['Value'] > test['Value'].median(), sarima_forecast > test['Value'].median())
    f1 = f1_score(test['Value'] > test['Value'].median(), sarima_forecast > test['Value'].median())

    st.markdown(f'<div class="metric">Mean Squared Error (MSE): {mse:.4f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric">Mean Absolute Error (MAE): {mae:.4f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric">Precision: {precision:.2f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric">Recall: {recall:.2f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric">F1 Score: {f1:.2f}</div>', unsafe_allow_html=True)

    # Visualizations with Altair
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    sarima_forecast_df = pd.DataFrame({'Date': test.index, 'Forecast': sarima_forecast})
    lstm_forecast_df = pd.DataFrame({'Date': df['Date'][-len(lstm_forecast):], 'Forecast': lstm_forecast.flatten()})

    chart_sarima = alt.Chart(sarima_forecast_df).mark_line(color='blue').encode(
        x='Date:T',
        y='Forecast:Q'
    ).properties(title='SARIMA Forecast')

    chart_lstm = alt.Chart(lstm_forecast_df).mark_line(color='red').encode(
        x='Date:T',
        y='Forecast:Q'
    ).properties(title='LSTM Forecast')

    st.altair_chart(chart_sarima, use_container_width=True)
    st.altair_chart(chart_lstm, use_container_width=True)

elif page == "Contact Us":
    st.markdown('<div class="title">Contact Us</div>', unsafe_allow_html=True)
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    st.markdown("""
        <form>
            <label for="name">Name:</label>
            <input type="text" id="name" name="name">

            <label for="email">Email:</label>
            <input type="email" id="email" name="email">

            <label for="message">Message:</label>
            <textarea id="message" name="message" rows="4"></textarea>

            <input type="submit" value="Submit">
        </form>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
