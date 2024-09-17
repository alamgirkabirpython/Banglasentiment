import streamlit as st
import pandas as pd
import os
import json
from Banglanlpdeeplearn.model import model_train
from Banglanlpdeeplearn.text_process import preprocess_text
from Banglanlpdeeplearn.predict import predict_sentiment
import bcrypt

import nltk
nltk.download('punkt', download_dir='/app/nltk_data')

nltk.data.path.append('/app/nltk_data')

# App logo and background images
logo_url = "https://img.freepik.com/free-vector/colorful-bird-illustration-gradient_343694-1741.jpg?size=626&ext=jpg&ga=GA1.1.733875022.1726100029&semt=ais_hybrid.png"
background_image_url = "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?crop=entropy&cs=tinysrgb&w=1080&fit=max"

# Set up Streamlit app configuration
st.set_page_config(page_title="Bangla Sentiment Analysis", page_icon=logo_url, layout="wide")

# Custom CSS for styling the app with background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .stButton>button {{
        background-color: #3498db;
        color: white;
        border-radius: 10px;
    }}
    .stButton>button:hover {{
        background-color: #2980b9;
    }}
    .stTextInput, .stTextArea {{
        background-color: rgba(255, 255, 255, 0.8);
    }}
    h1, h2, h3 {{
        color: white;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }}
    .centered {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
    }}
    .dataframe-container {{
        width: 80%;
        margin: 0 auto;
        height: 150px;
        overflow-y: auto;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load users from a JSON file
users_file = 'users.json'

def load_users():
    if os.path.exists(users_file):
        with open(users_file, 'r') as file:
            return json.load(file)
    else:
        return {}

# Save users to a JSON file
def save_users(users_db):
    with open(users_file, 'w') as file:
        json.dump(users_db, file)

# Load user database from file
users_db = load_users()

# Caching the data loading function
@st.cache_data
def data_load(file_path):
    df = pd.read_csv(file_path)
    if df.isnull().values.any():
        st.warning("Warning: The dataset contains missing values. Please clean the data.")
    df['processed_text'] = df['text'].apply(preprocess_text)
    return df

# Caching the model training function
@st.cache_resource
def train_model(file_path):
    df = data_load(file_path)
    model1, model2, tokenizer, encoder, X_test, y_test, max_length = model_train(df, 'processed_text', 'label')
    return model1, model2, tokenizer, encoder, X_test, y_test, max_length

# Load the dataset and train models
file_path = "https://raw.githubusercontent.com/alamgirkabirpython/Banglasentiment/17d631b16b6e920ed8a2c8057a544ebb4e5d81db/bangla_sentiment_data.csv"
model1, model2, tokenizer, encoder, X_test, y_test, max_length = train_model(file_path)

# Function to hash passwords before storing them
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# Function to verify hashed passwords
def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# Display the app logo
st.image(logo_url, width=100)

# Main page title
st.title("Bangla Sentiment Analysis :heart_eyes: :cry:")

# Sidebar with example input texts
with st.sidebar:
    st.write("### Input Text Example:")
    st.code("আমি খুব খারাপ আছি 😢", language="plain")
    st.code("তুমি কেন এমন করলে? 😡", language="plain")
    st.code("তুমি আমাকে রাগিয়ে দিচ্ছ 😠", language="plain")
    st.code("আজকে সবকিছুই বিরক্তিকর 😡", language="plain")
    st.code("আজ আমার খুব মন খারাপ।", language="plain")
    st.code("আমি খুব একা বোধ করছি।", language="plain")
    st.code("জীবনটা কেন এত কঠিন!", language="plain")
    st.code("কিছুই যেন আর ভালো লাগছে না।", language="plain")
    st.code("মনে হচ্ছে সব কিছু ভেঙে পড়ছে।", language="plain")
    st.code("আজ আমি খুব আনন্দিত!", language="plain")
    st.code("এটা আমার জীবনের সেরা মুহূর্ত!", language="plain")
    st.code("সবকিছু এত সুন্দর লাগছে!", language="plain")
    st.code("আজকের দিনটা সত্যিই অসাধারণ!", language="plain")

# Create navigation buttons
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    login_button = st.button("Login", key="login_button")
with col2:
    signup_button = st.button("Sign Up", key="signup_button")
with col3:
    data_button = st.button("Data", key="data_button")
with col4:
    contact_button = st.button("Contact Information", key="contact_button")

# Manage session state for navigation
if "selected_option" not in st.session_state:
    st.session_state["selected_option"] = None

if login_button:
    st.session_state["selected_option"] = "Login"
elif signup_button:
    st.session_state["selected_option"] = "Sign Up"
elif data_button:
    st.session_state["selected_option"] = "Data"
elif contact_button:
    st.session_state["selected_option"] = "Contact Information"

# Handle user authentication (Login and Signup)
if st.session_state["selected_option"] == "Login":
    st.subheader("Login")
    login_username = st.text_input("Username", key="login_username_input")
    login_password = st.text_input("Password", type="password", key="login_password_input")
    if st.button("Login", key="login_confirm_button"):
        if login_username in users_db and check_password(login_password, users_db[login_username]):
            st.session_state["authenticated"] = True
            st.session_state["username"] = login_username
            st.success(f"Welcome {login_username}! You are now logged in.")
        else:
            st.error("Invalid username or password.")

elif st.session_state["selected_option"] == "Sign Up":
    st.subheader("Sign Up")
    new_username = st.text_input("Choose a Username", key="signup_username_input")
    new_password = st.text_input("Choose a Password", type="password", key="signup_password_input")
    if st.button("Sign Up", key="signup_confirm_button"):
        if new_username in users_db:
            st.warning("Username already exists. Please choose a different username.")
        elif new_username and new_password:
            users_db[new_username] = hash_password(new_password)
            save_users(users_db)
            st.success("Sign up successful! You can now log in.")
        else:
            st.error("Please fill both fields.")

# Handle data and model display
elif st.session_state["selected_option"] == "Data":
    st.subheader("Data")
    if "authenticated" in st.session_state and st.session_state["authenticated"]:
        df = data_load(file_path)

        # Center the DataFrame
        st.markdown(
            """
            <div class="centered">
                <div class="dataframe-container">
                    {dataframe}
                </div>
            </div>
            """.format(dataframe=df.to_html(index=False, escape=False)),
            unsafe_allow_html=True
        )

        # Show model performance
        st.write("### Model Performance")
        model_choice = st.selectbox("Choose a model:", ['Model 1', 'Model 2'])
        model = model1 if model_choice == 'Model 1' else model2

        if model:
            loss, accuracy = model.evaluate(X_test, y_test)
            st.metric(label="Accuracy", value=f"{accuracy:.2f}")
            st.progress(accuracy)

        # Sentiment prediction
        st.write("## Predict Sentiment")
        user_input = st.text_area("Enter Bangla text for prediction", "")
        if st.button("Show Prediction", key="predict_button"):
            if user_input:
                predicted_label = predict_sentiment(user_input, model, tokenizer, encoder, max_length)
                st.success(f"Predicted Sentiment: {predicted_label}")
            else:
                st.warning("Please enter some text for prediction.")

elif st.session_state["selected_option"] == "Contact Information":
    st.subheader("Contact Information")
    st.write("Email: your-email@example.com")
    st.write("Phone: +880-1234567890")
