import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import math

# Custom CSS for the app
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
            color: #333;
        }
        .css-18e3th9 {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stApp {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #2c3e50;
        }
        h2 {
            color: #34495e;
        }
        .stButton button {
            background-color: #3498db;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            border: none;
        }
        .stRadio > div {
            flex-direction: row;
            justify-content: center;
        }
        .stRadio label {
            padding-right: 1rem;
            font-size: 1.2rem;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)

# Function to create the About Us page
def about_us():
    st.title("About Us")

    st.markdown("""
    ## Our Mission

    At **General Index Forecasting**, our mission is to provide accurate and insightful forecasts using advanced machine learning and statistical models. By leveraging the power of LSTM (Long Short-Term Memory) and SARIMA (Seasonal Autoregressive Integrated Moving Average), we aim to help businesses and individuals make data-driven decisions to navigate future trends with confidence.

    ## The Project

    This project focuses on forecasting the General Index over a 5-year period. By analyzing historical data and applying state-of-the-art predictive models, we generate forecasts that can guide strategic planning and decision-making.

    **Key Features:**
    - **LSTM Model:** A deep learning approach that captures complex patterns in time series data.
    - **SARIMA Model:** A traditional statistical method known for its effectiveness in seasonal data forecasting.
    - **Comparison of Models:** We provide a side-by-side comparison of LSTM and SARIMA forecasts, allowing users to evaluate the strengths of each model.

    ## Our Team

    This project was developed by a team of passionate data scientists and engineers dedicated to pushing the boundaries of what's possible with predictive analytics.

    **Team Members:**
    - **[Your Name]:** Lead Data Scientist & Developer
    - **[Collaborator's Name]:** Machine Learning Engineer
    - **[Collaborator's Name]:** Data Analyst

    ## Contact Us

    We are always open to collaboration and feedback. If you have any questions or suggestions, feel free to reach out to us at [your-email@example.com].

    ## Future Plans

    We are committed to continuously improving our models and expanding the project to cover more indices and economic indicators. Stay tuned for more updates and features!

    **Thank you for using our application!**
    """)

# Function to run the forecasting app
def run_forecasting_app():
    # Streamlit App Title
    st.title('General Index Forecasting using LSTM and SARIMA')

    # Load the dataset
    file_path = st.text_input('Enter the path to the cleaned data CSV file', 'cleaned_data.csv')
    data = pd.read_csv(file_path)

    # Display the DataFrame
    st.write("Data Preview:")
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

    # Calculate metrics
    mse_sarima = mean_squared_error(data['General index'][-forecast_steps:], forecast_mean_sarima[-forecast_steps:])
    rmse_sarima = math.sqrt(mse_sarima)

    mse_lstm = mean_squared_error(data['General index'][-forecast_steps:], future_predictions_lstm_inv[-forecast_steps:])
    rmse_lstm = math.sqrt(mse_lstm)

    # Prepare data for plotting SARIMA and LSTM forecasts
    forecast_data_sarima = pd.DataFrame({
        'Year': forecast_index_sarima.year,
        'Forecasted General Index (SARIMA)': forecast_mean_sarima
    })

    forecast_data_lstm = pd.DataFrame({
        'Year': future_dates_lstm.year,
        'Forecasted General Index (LSTM)': future_predictions_lstm_inv.flatten()
    })

    # Separate Plotting for SARIMA
    st.subheader('SARIMA Forecast')
    sarima_chart = alt.Chart(forecast_data_sarima).mark_line(color='blue').encode(
        x=alt.X('Year:O', title='Year'),
        y='Forecasted General Index (SARIMA):Q',
        tooltip=['Year:O', 'Forecasted General Index (SARIMA):Q']
    ).properties(
        width=700,
        height=400
    )
    st.altair_chart(sarima_chart)

    # Separate Plotting for LSTM
    st.subheader('LSTM Forecast')
    lstm_chart = alt.Chart(forecast_data_lstm).mark_line(color='green').encode(
        x=alt.X('Year:O', title='Year'),
        y='Forecasted General Index (LSTM):Q',
        tooltip=['Year:O', 'Forecasted General Index (LSTM):Q']
    ).properties(
        width=700,
        height=400
    )
    st.altair_chart(lstm_chart)

    # Combined Forecast Plot
    st.subheader('Combined Forecast Comparison')
    combined_data = pd.DataFrame({
        'Year': forecast_index_sarima.year,
        'SARIMA': forecast_mean_sarima,
        'LSTM': future_predictions_lstm_inv.flatten()
    })
    combined_chart = alt.Chart(combined_data).transform_fold(
        ['SARIMA', 'LSTM'],
        as_=['Model', 'Forecasted General Index']
    ).mark_line().encode(
        x=alt.X('Year:O', title='Year'),
        y='Forecasted General Index:Q',
        color='Model:N',
        tooltip=['Year:O', 'Model:N', 'Forecasted General Index:Q']
    ).properties(
        width=700,
        height=400
    )
    st.altair_chart(combined_chart)

    # Display MSE and RMSE for each model
    st.subheader('Model Evaluation Metrics')
    st.write(f"SARIMA - MSE: {mse_sarima}, RMSE: {rmse_sarima}")
    st.write(f"LSTM - MSE: {mse_lstm}, RMSE: {rmse_lstm}")

# Add navigation to the top
page = st.radio("Navigate to:", ["About Us", "Forecasting Models"], horizontal=True)

# Render the selected page
if page == "About Us":
    about_us()
elif page == "Forecasting Models":
    run_forecasting_app()
