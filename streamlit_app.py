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
image_path = "inflation3.jpg"

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
        padding: 10px;
        background-color: rgba(255, 255, 255, 0.8); /* Semi-transparent background for readability */
    }}
    .subheader {{
        font-size: 24px;
        color: #2196F3;
        margin-top: 20px;
        margin-bottom: 10px;
        background-color: rgba(255, 255, 255, 0.8);
        padding: 5px;
    }}
    .metric {{
        font-size: 18px;
        font-weight: bold;
        color: #FF5722;
        background-color: rgba(255, 255, 255, 0.8);
        padding: 5px;
        border-radius: 5px;
    }}
    .about, .contact {{
        background-color: rgba(255, 255, 255, 0.8);
        padding: 15px;
        border-radius: 8px;
        margin: 15px;
    }}
    .vertical-radio input[type="radio"] {{
        display: block;
        margin: 10px 0;
    }}
    </style>
""", unsafe_allow_html=True)

# Radio buttons for navigation in vertical layout
st.markdown("""
    <div class="vertical-radio">
        <label><input type="radio" name="page" value="About Us"> About Us</label>
        <label><input type="radio" name="page" value="Home" checked> Home</label>
        <label><input type="radio" name="page" value="Contact Us"> Contact Us</label>
    </div>
""", unsafe_allow_html=True)

# Get selected page from the radio buttons
page_selection = st.radio("Choose a page:", ["About Us", "Home", "Contact Us"], index=1)

if page_selection == "About Us":
    st.markdown('<div class="title">About Us</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='about'>
    <h2>Welcome to Our Forecasting App!</h2>
    <p>Our app provides insights into forecasting economic indices using advanced machine learning models. We leverage the power of Long Short-Term Memory (LSTM) networks and Seasonal Autoregressive Integrated Moving Average (SARIMA) models to deliver accurate forecasts and valuable metrics.</p>
    <p><strong>Mission:</strong> To enhance decision-making with data-driven insights and advanced forecasting techniques.</p>
    <p><strong>Vision:</strong> To be at the forefront of predictive analytics and contribute to solving real-world problems through innovative technologies.</p>
    <p>Feel free to explore the "Home" page to see our forecasting models in action and the "About Us" page to learn more about our mission and vision.</p>
    <p>Thank you for visiting!</p>
    </div>
    """, unsafe_allow_html=True)

elif page_selection == "Home":
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
        width=700,
        height=400
    ).interactive()
    st.altair_chart(base_chart)

    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Creating the dataset for LSTM
    def create_dataset(dataset, time_step=1):
        X, Y = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]
            X.append(a)
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)

    time_step = 12
    X, Y = create_dataset(scaled_data, time_step)

    # Reshape input to be [samples, time steps, features] for LSTM
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split the data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    st.subheader('Training LSTM Model...')
    model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test), verbose=1)

    # Predicting the next 60 months (5 years) using LSTM
    forecast_steps = 60
    future_predictions_lstm = []

    current_input_lstm = X_test[-1].reshape(1, time_step, 1)
    for _ in range(forecast_steps):
        future_pred_lstm = model.predict(current_input_lstm)
        future_predictions_lstm.append(future_pred_lstm[0, 0])
        current_input_lstm = np.append(current_input_lstm[:, 1:, :], future_pred_lstm.reshape(1, 1, 1), axis=1)

    future_dates_lstm = pd.date_range(data.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='M')
    future_predictions_lstm_inv = scaler.inverse_transform(np.array(future_predictions_lstm).reshape(-1, 1))

    # Define the SARIMA model
    sarima_model = SARIMAX(data['General index'], 
                           order=(1, 1, 1),  # ARIMA parameters (p, d, q)
                           seasonal_order=(1, 1, 1, 12),  # Seasonal parameters (P, D, Q, s)
                           enforce_stationarity=False,
                           enforce_invertibility=False)

    # Fit the model
    sarima_results = sarima_model.fit(disp=False)

    # Forecasting the next 60 months (5 years) using SARIMA
    forecast_sarima = sarima_results.get_forecast(steps=forecast_steps)
    forecast_index_sarima = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='M')
    forecast_mean_sarima = forecast_sarima.predicted_mean
    forecast_conf_int_sarima = forecast_sarima.conf_int()

    # Dummy future actual values for comparison (Replace with actual future values if available)
    dummy_future_actual = np.random.rand(forecast_steps)  # Replace with actual future values

    # Convert predictions to binary (using a threshold)
    threshold = 0.5
    lstm_binary_preds = (future_predictions_lstm_inv.flatten() >= threshold).astype(int)
    sarima_binary_preds = (forecast_mean_sarima >= threshold).astype(int)
    dummy_binary_actual = (dummy_future_actual >= threshold).astype(int)

    # Evaluate SARIMA
    precision_sarima = precision_score(dummy_binary_actual, sarima_binary_preds)
    recall_sarima = recall_score(dummy_binary_actual, sarima_binary_preds)
    f1_sarima = f1_score(dummy_binary_actual, sarima_binary_preds)
    accuracy_sarima = accuracy_score(dummy_binary_actual, sarima_binary_preds)
    mse_sarima = mean_squared_error(dummy_future_actual, forecast_mean_sarima)
    rmse_sarima = np.sqrt(mse_sarima)

    # Plot SARIMA and LSTM forecasts
    st.subheader('Forecast Comparison: SARIMA vs LSTM')

    # SARIMA Plot
    sarima_chart = alt.Chart(pd.DataFrame({
        'Date': forecast_index_sarima,
        'Forecast': forecast_mean_sarima
    })).mark_line(color='blue').encode(
        x='Date:T',
        y='Forecast:Q'
    ).properties(
        width=700,
        height=400
    ).interactive()

    # LSTM Plot
    lstm_chart = alt.Chart(pd.DataFrame({
        'Date': future_dates_lstm,
        'Forecast': future_predictions_lstm_inv.flatten()
    })).mark_line(color='red').encode(
        x='Date:T',
        y='Forecast:Q'
    ).properties(
        width=700,
        height=400
    ).interactive()

    # Comparison Plot
    combined_chart = alt.layer(sarima_chart, lstm_chart).resolve_scale(y='shared')
    st.altair_chart(combined_chart)

    st.subheader('Evaluation Metrics for SARIMA:')
    st.write(f'**Mean Squared Error (MSE):** {mse_sarima}')
    st.write(f'**Root Mean Squared Error (RMSE):** {rmse_sarima}')
    st.write(f'**Precision:** {precision_sarima}')
    st.write(f'**Recall:** {recall_sarima}')
    st.write(f'**F1 Score:** {f1_sarima}')
    st.write(f'**Accuracy:** {accuracy_sarima}')

elif page_selection == "Contact Us":
    st.markdown('<div class="title">Contact Us</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='contact'>
    <h2>Get in Touch!</h2>
    <p>We would love to hear from you. If you have any questions or feedback, please reach out to us using the contact details below:</p>
    <p><strong>Email:</strong> contact@forecastingapp.com</p>
    <p><strong>Phone:</strong> +1 (123) 456-7890</p>
    <p><strong>Address:</strong> 123 Data Drive, Analytics City, AC 12345</p>
    <p>Thank you for your interest in our app. We look forward to connecting with you!</p>
    </div>
    """, unsafe_allow_html=True)
