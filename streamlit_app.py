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

# Create the CSS with the Base64 encoded image and enhanced styling
st.markdown(f"""
    <style>
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    .main {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        color: #ffffff;
        font-family: 'Arial', sans-serif;
        animation: fadeIn 1s ease-in-out;
    }}
    .title {{
        font-size: 42px;
        color: #ffffff;
        text-align: center;
        padding: 20px;
        background: linear-gradient(to right, #ff7e5f, #feb47b);
        border-radius: 12px;
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
        animation: fadeIn 1.5s ease-in-out;
    }}
    .subheader {{
        font-size: 28px;
        color: #ffffff;
        margin-top: 20px;
        margin-bottom: 20px;
        background: linear-gradient(to right, #00c6ff, #0072ff);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        animation: fadeIn 2s ease-in-out;
    }}
    .metric {{
        font-size: 20px;
        font-weight: bold;
        color: #ffffff;
        background: linear-gradient(to right, #ff6a00, #ee0979);
        padding: 12px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        text-align: center;
        margin-bottom: 20px;
        animation: fadeIn 2.5s ease-in-out;
    }}
    .nav {{
        display: flex;
        justify-content: center;
        margin-bottom: 30px;
    }}
    .nav input[type="radio"] {{
        display: none;
    }}
    .nav label {{
        background: linear-gradient(to right, #ff7e5f, #feb47b);
        color: #ffffff;
        padding: 15px 30px;
        border-radius: 10px;
        cursor: pointer;
        margin: 0 8px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        font-weight: bold;
        transition: background 0.3s ease, transform 0.3s ease;
    }}
    .nav input[type="radio"]:checked + label {{
        background: linear-gradient(to right, #333333, #616161);
        transform: scale(1.05);
    }}
    .content {{
        font-size: 24px;
        line-height: 1.8;
        color: #ffffff;
        background: rgba(0, 0, 0, 0.7);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        animation: fadeIn 3s ease-in-out;
    }}
    .content h2 {{
        font-size: 36px;
        color: #ff7e5f;
        text-align: center;
    }}
    .content h3 {{
        font-size: 30px;
        color: #00c6ff;
    }}
    .content ul {{
        font-size: 22px;
        margin: 15px 0;
        padding: 0;
        list-style: square;
    }}
    .content ul li {{
        margin-bottom: 12px;
    }}
    .content a {{
        color: #00c6ff;
        text-decoration: none;
        font-weight: bold;
    }}
    .content a:hover {{
        text-decoration: underline;
    }}
    .form-container {{
        font-size: 24px;
        line-height: 1.8;
        color: #ffffff;
        background: rgba(0, 0, 0, 0.7);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        animation: fadeIn 3s ease-in-out;
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
        margin-top: 10px;
        border: 1px solid #dddddd;
        border-radius: 10px;
        font-size: 20px;
    }}
    .form-container input[type="submit"] {{
        background: linear-gradient(to right, #ff7e5f, #feb47b);
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
    <div class="content">
    <h2>Welcome to General Index Forecasting!</h2>
    <p>We are a dedicated team of data scientists and analysts passionate about harnessing the power of data to drive informed decisions. Our mission is to provide actionable insights through advanced forecasting techniques and data-driven analysis.</p>
    
    <h3>Our Expertise:</h3>
    <ul>
        <li><strong>Data Analytics:</strong> Transforming raw data into meaningful insights.</li>
        <li><strong>Machine Learning:</strong> Developing predictive models to anticipate future trends.</li>
        <li><strong>Business Intelligence:</strong> Leveraging data to enhance business strategies.</li>
    </ul>
    
    <h3>Our Collaboration with MOSPI:</h3>
    <p>This project is specifically designed to cater to the needs of the Ministry of Statistics and Programme Implementation (MOSPI). We aim to provide MOSPI with advanced forecasting tools and insights to support their data-driven decision-making processes.</p>
    
    <h3>Our Team:</h3>
    <ul>
        <li><strong>Sai Manasa B</strong> - Data Scientist: With over 4 years of experience in data science, Sai specializes in developing machine learning models and data analysis.</li>
        <li><strong>Kimberly Marclin Nathan A</strong> - Data Analyst: Kim has a knack for turning complex data into clear and actionable insights, integrating data analytics into business strategies to drive growth and efficiency.</li>
    </ul>
    
    <h3>Our Vision:</h3>
    <p>We aim to be at the forefront of data science and analytics, continuously innovating to provide our clients, including MOSPI, with the best tools and strategies for success. Our goal is to empower organizations with the knowledge and foresight to make data-driven decisions.</p>
    
    <h3>Our Values:</h3>
    <ul>
        <li><strong>Integrity:</strong> We uphold the highest standards of honesty and transparency.</li>
        <li><strong>Innovation:</strong> We embrace creativity and new ideas to solve complex problems.</li>
        <li><strong>Excellence:</strong> We are committed to delivering high-quality results and solutions.</li>
    </ul>
    
    <h3>Get Involved:</h3>
    <p>We are always looking to collaborate with like-minded professionals and organizations. If you're interested in working with us or have any questions, feel free to reach out through our <a href="#Contact Us">Contact Us</a> page.</p>
    
    <h3>Thank You!</h3>
    <p>We appreciate your interest in our work and look forward to the opportunity to collaborate with you.</p>
    </div>
    """, unsafe_allow_html=True)
    
elif page == "Home":
    st.markdown('<div class="title">Home - General Index Forecasting</div>', unsafe_allow_html=True)

    # Load and preview dataset
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())

        # Ensure required columns exist in the dataset
        if 'DATE' in df.columns and 'General Index' in df.columns:
            # Process the data
            df['DATE'] = pd.to_datetime(df['DATE'])
            df.set_index('DATE', inplace=True)

            # Data scaling
            scaler = MinMaxScaler()
            df['General Index_scaled'] = scaler.fit_transform(df[['General Index']])

            # Altair visualization of the General Index
            st.markdown('<div class="subheader">Visualizing General Index Over Time</div>', unsafe_allow_html=True)
            base = alt.Chart(df.reset_index()).mark_line().encode(
                x=alt.X('DATE:T', title='Year'),
                y=alt.Y('General Index:Q', title='General Index'),
                tooltip=['DATE:T', 'General Index:Q']
            ).properties(
                width=700,
                height=400
            ).interactive()

            st.altair_chart(base)

            # LSTM Model training and forecasting
            def create_lstm_model(input_shape):
                model = Sequential()
                model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
                model.add(Dropout(0.2))
                model.add(LSTM(50, return_sequences=False))
                model.add(Dropout(0.2))
                model.add(Dense(25))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                return model

            st.markdown('<div class="subheader">LSTM Model Training</div>', unsafe_allow_html=True)

            # Split data into training and test sets
            train_size = int(len(df) * 0.8)
            train_data = df.iloc[:train_size]['General Index_scaled'].values
            test_data = df.iloc[train_size:]['General Index_scaled'].values

            # Prepare the data for LSTM model
            def prepare_lstm_data(data, time_step=1):
                X, y = [], []
                for i in range(len(data) - time_step - 1):
                    a = data[i:(i + time_step)]
                    X.append(a)
                    y.append(data[i + time_step])
                return np.array(X), np.array(y)

            time_step = 10
            X_train, y_train = prepare_lstm_data(train_data, time_step)
            X_test, y_test = prepare_lstm_data(test_data, time_step)

            # Reshape input to be [samples, time steps, features]
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # Create LSTM model
            model = create_lstm_model((X_train.shape[1], 1))

            # Train the model
            model.fit(X_train, y_train, batch_size=64, epochs=50, verbose=1)

            # Forecasting with LSTM
            train_predict = model.predict(X_train)
            test_predict = model.predict(X_test)

            # Inverse transform to get actual values
            train_predict = scaler.inverse_transform(train_predict)
            test_predict = scaler.inverse_transform(test_predict)
            y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
            y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

            # Evaluate the model
            train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))

            st.markdown(f'<div class="metric">LSTM Train RMSE: {train_rmse:.2f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric">LSTM Test RMSE: {test_rmse:.2f}</div>', unsafe_allow_html=True)

            # SARIMA model for comparison
            st.markdown('<div class="subheader">SARIMA Model for Comparison</div>', unsafe_allow_html=True)

            sarima_model = SARIMAX(df['General Index'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            sarima_result = sarima_model.fit(disp=False)
            sarima_forecast = sarima_result.forecast(steps=len(test_data))

            sarima_rmse = np.sqrt(mean_squared_error(df.iloc[-len(test_data):]['General Index'], sarima_forecast))

            st.markdown(f'<div class="metric">SARIMA Test RMSE: {sarima_rmse:.2f}</div>', unsafe_allow_html=True)

            # Plotting LSTM and SARIMA forecasts
            st.markdown('<div class="subheader">Comparison of LSTM and SARIMA Forecasts</div>', unsafe_allow_html=True)

            lstm_df = df.iloc[train_size + time_step + 1:].copy()
            lstm_df['LSTM_Forecast'] = test_predict

            sarima_df = df.iloc[-len(test_data):].copy()
            sarima_df['SARIMA_Forecast'] = sarima_forecast.values

            lstm_chart = alt.Chart(lstm_df.reset_index()).mark_line(color='orange').encode(
                x='DATE:T',
                y='LSTM_Forecast:Q',
                tooltip=['DATE:T', 'LSTM_Forecast:Q']
            ).properties(
                width=700,
                height=400
            )

            sarima_chart = alt.Chart(sarima_df.reset_index()).mark_line(color='blue').encode(
                x='DATE:T',
                y='SARIMA_Forecast:Q',
                tooltip=['DATE:T', 'SARIMA_Forecast:Q']
            ).properties(
                width=700,
                height=400
            )

            combined_chart = alt.layer(base, lstm_chart, sarima_chart).resolve_scale(y='independent')
            st.altair_chart(combined_chart)

        else:
            st.warning("Dataset does not contain required columns 'DATE' and 'General Index'.")

elif page == "Contact Us":
    st.markdown('<div class="title">Contact Us</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="form-container">
        <form action="https://formspree.io/f/{your_form_id}" method="POST">
            <label for="name">Name:</label>
            <input type="text" id="name" name="name" required>

            <label for="email">Email:</label>
            <input type="email" id="email" name="email" required>

            <label for="message">Message:</label>
            <textarea id="message" name="message" rows="5" required></textarea>

            <input type="submit" value="Send">
        </form>
    </div>
    """, unsafe_allow_html=True)
