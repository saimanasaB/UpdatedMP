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

# Create the CSS with the Base64 encoded image
# Updated CSS with more attractive styling for 'About Us' and 'Contact Us' pages
# Updated CSS with more attractive styling for 'About Us' and 'Contact Us' pages
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
        color: #ffffff;
        text-align: center;
        padding: 20px;
        background: linear-gradient(to right, #4CAF50, #81C784);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .subheader {{
        font-size: 24px;
        color: #ffffff;
        margin-top: 20px;
        margin-bottom: 10px;
        background: linear-gradient(to right, #2196F3, #64B5F6);
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }}
    .metric {{
        font-size: 18px;
        font-weight: bold;
        color: #ffffff;
        background: linear-gradient(to right, #FF5722, #FF8A65);
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
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
        background: linear-gradient(to right, #4CAF50, #81C784);
        color: #ffffff;
        padding: 12px 24px;
        border-radius: 8px;
        cursor: pointer;
        margin: 0 5px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        font-weight: bold;
        transition: background 0.3s ease;
    }}
    .nav input[type="radio"]:checked + label {{
        background: linear-gradient(to right, #333333, #616161);
    }}
    .content {{
        font-size: 22px;
        line-height: 1.8;
        color: #333;
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
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
        background: linear-gradient(to right, #4CAF50, #81C784);
        color: #ffffff;
        padding: 12px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 20px;
        font-weight: bold;
        transition: background 0.3s ease;
    }}
    .form-container input[type="submit"]:hover {{
        background: linear-gradient(to right, #333333, #616161);
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
    
    # Evaluate LSTM
    precision_lstm = precision_score(dummy_binary_actual, lstm_binary_preds)
    recall_lstm = recall_score(dummy_binary_actual, lstm_binary_preds)
    f1_lstm = f1_score(dummy_binary_actual, lstm_binary_preds)
    accuracy_lstm = accuracy_score(dummy_binary_actual, lstm_binary_preds)
    mse_lstm = mean_squared_error(dummy_future_actual, future_predictions_lstm_inv.flatten())
    rmse_lstm = np.sqrt(mse_lstm)
    
    st.subheader('Model Evaluation Metrics')
    st.write(f"<div class='metric'>SARIMA - Precision: {precision_sarima}, Recall: {recall_sarima}, F1 Score: {f1_sarima}, Accuracy: {accuracy_sarima}, MSE: {mse_sarima}, RMSE: {rmse_sarima}</div>", unsafe_allow_html=True)
    st.write(f"<div class='metric'>LSTM - Precision: {precision_lstm}, Recall: {recall_lstm}, F1 Score: {f1_lstm}, Accuracy: {accuracy_lstm}, MSE: {mse_lstm}, RMSE: {rmse_lstm}</div>", unsafe_allow_html=True)
    
    # Prepare data for plotting SARIMA and LSTM forecasts
    forecast_data_sarima = pd.DataFrame({
        'Date': forecast_index_sarima,
        'Year': forecast_index_sarima.year,
        'Forecasted General Index (SARIMA)': forecast_mean_sarima
    })
    
    forecast_data_lstm = pd.DataFrame({
        'Date': future_dates_lstm,
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
    
    # Comparison of forecasts
    comparison_data = pd.concat([
        forecast_data_sarima[['Year', 'Forecasted General Index (SARIMA)']].rename(columns={'Forecasted General Index (SARIMA)': 'Forecast', 'Year': 'Year'}).assign(Model='SARIMA'),
        forecast_data_lstm[['Year', 'Forecasted General Index (LSTM)']].rename(columns={'Forecasted General Index (LSTM)': 'Forecast', 'Year': 'Year'}).assign(Model='LSTM')
    ])
    
    comparison_chart = alt.Chart(comparison_data).mark_line().encode(
        x=alt.X('Year:O', title='Year'),
        y=alt.Y('Forecast:Q', title='Forecasted General Index'),
        color='Model:N',
        tooltip=['Year:O', 'Model:N', 'Forecast:Q']
    ).properties(
        width=700,
        height=400
    )
    st.altair_chart(comparison_chart)
    
    # Ensure the plots and metrics are displayed properly
    st.subheader('Forecast Data')
    st.write("Forecasted General Index using SARIMA:")
    st.dataframe(forecast_data_sarima)
    
    st.write("Forecasted General Index using LSTM:")
    st.dataframe(forecast_data_lstm)

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
