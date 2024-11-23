import streamlit as st
from utils.css_loader import load_css
from utils.model_training import train_model
from utils.file_upload import upload_and_predict 
from utils.detect_emotion import detect_emotion

st.set_page_config(
    page_title="Sentiment Analysis Web App", 
    page_icon=":smiley:",  # Smiley emoji to represent emotions
    layout="wide")
def render():
    load_css("style.css")

    # HTML content
    html_content = """
    <div class="title-container">
        <h1>
        <span><span>ML Sentiment</span> <span>Detection Model</span></span>
        </h1>
    </div>
    <p style="text-align: center; font-size: 1.5em; color: #d5dbdb;">Welcome to Emotion Detection Web App: Your Digital Companion</p>
    """
    st.markdown(html_content, unsafe_allow_html=True)

    train_model()  # Train the model if needed
    detect_emotion()
    upload_and_predict()  # File upload and prediction
