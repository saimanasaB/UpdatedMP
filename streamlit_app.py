import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
    .nav {{
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
        flex-direction: row;
    }}
    .nav button {{
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin: 0 5px;
        text-align: center;
    }}
    .nav button.active {{
        background-color: #333;
    }}
    </style>
""", unsafe_allow_html=True)

# Page navigation
page = st.session_state.get('page', 'Home')

def update_page(page_name):
    st.session_state['page'] = page_name

st.markdown('<div class="nav">', unsafe_allow_html=True)
if st.button('About Us', key='about_us'):
    update_page('About Us')
if st.button('Home', key='home'):
    update_page('Home')
if st.button('Contact Us', key='contact_us'):
    update_page('Contact Us')
st.markdown('</div>', unsafe_allow_html=True)

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
    <li><strong>John Doe</strong> - Lead Data Scientist: With over 10 years of experience in data science, John specializes in developing machine learning models and data analysis.</li>
    <li><strong>Jane Smith</strong> - Data Analyst: Jane has a knack for turning complex data into clear and actionable insights.</li>
    <li><strong>Alex Johnson</strong> - Business Intelligence Specialist: Alex focuses on integrating data analytics into business strategies to drive growth and efficiency.</li>
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
    history = model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_test, Y_test), verbose=1)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse transform the predictions and actual values
    predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], data.shape[1]-1))), axis=1))[:,0]
    Y_test_inv = scaler.inverse_transform(np.concatenate((Y_test.reshape(-1, 1), np.zeros((Y_test.shape[0], data.shape[1]-1))), axis=1))[:,0]
    
    # Plot the predictions
    st.subheader('LSTM Model Predictions vs Actual Values')
    predictions_df = pd.DataFrame({
        'Date': data.index[-len(predictions):],
        'Actual': Y_test_inv,
        'Predicted': predictions
    })
    
    chart = alt.Chart(predictions_df).mark_line().encode(
        x='Date:T',
        y='Actual:Q',
        color=alt.value('red')
    ).properties(
        width=700,
        height=400
    )
    
    predicted_line = alt.Chart(predictions_df).mark_line().encode(
        x='Date:T',
        y='Predicted:Q',
        color=alt.value('blue')
    )
    
    st.altair_chart(chart + predicted_line)
    
    # Model evaluation
    st.subheader('Model Evaluation Metrics')
    mse = mean_squared_error(Y_test_inv, predictions)
    mae = mean_absolute_error(Y_test_inv, predictions)
    
    st.markdown(f"""
    **Mean Squared Error (MSE):** {mse:.2f}
    **Mean Absolute Error (MAE):** {mae:.2f}
    """)
    
elif page == "Contact Us":
    st.markdown('<div class="title">Contact Us</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='font-size: 22px; line-height: 1.6; color: #333; background-color: rgba(255, 255, 255, 0.9); padding: 15px; border-radius: 8px;'>
    <h2 style='font-size: 28px; color: #4CAF50;'>Get in Touch</h2>

    We would love to hear from you! Whether you have questions, feedback, or collaboration opportunities, feel free to reach out to us.

    <h3 style='font-size: 22px; color: #2196F3;'>Contact Information:</h3>
    <p><strong>Email:</strong> <a href="mailto:contact@forecasting.com" style="color: #2196F3;">contact@forecasting.com</a></p>
    <p><strong>Phone:</strong> +1-234-567-890</p>

    <h3 style='font-size: 22px; color: #2196F3;'>Follow Us:</h3>
    <p>Stay connected with us on social media for the latest updates and insights:</p>
    <ul style='font-size: 20px;'>
    <li><a href="https://twitter.com/forecasting" style="color: #2196F3;">Twitter</a></li>
    <li><a href="https://facebook.com/forecasting" style="color: #2196F3;">Facebook</a></li>
    <li><a href="https://linkedin.com/company/forecasting" style="color: #2196F3;">LinkedIn</a></li>
    </ul>

    <h3 style='font-size: 22px; color: #2196F3;'>Office Address:</h3>
    <p>
    General Index Forecasting<br>
    123 Data Street,<br>
    Suite 456,<br>
    City, State, ZIP Code,<br>
    Country
    </p>

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

    </div>
    """, unsafe_allow_html=True)
