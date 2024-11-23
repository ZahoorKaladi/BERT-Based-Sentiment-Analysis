import streamlit as st
from PIL import Image

def render():
    # Add the title and introductory section
    st.title("About Us")
    st.write("""
        Welcome to the BERT-based Sentimental Analysis Web App! 
        This project is developed as personal Project**.
    """)
    

    # Add a project description
    st.header("Project Overview")
    st.write("""
        This project uses **BERT (Bidirectional Encoder Representations from Transformers)**, a state-of-the-art 
        natural language processing model, to detect emotions from textual data. 
        It analyzes user inputs and predicts emotional categories like happiness, sadness, anger, and more.
        
        ### Key Features:
        - **Real-time Emotion Detection**: Upload text files or write custom text, and the model will analyze and predict the emotion.
        - **Interactive Visualizations**: View your results through interactive graphs and charts.
        - **Scalable Model**: The project is built to handle large datasets and generate accurate predictions efficiently.
    """)
    
    # Add a section on developer information
    st.header("Meet the Developer")
    st.write("""
        **Zahoor Ahmed** is a  Computer Science graduate at **University of Sindh**.
        I have a keen interest in Machine Learning and Natural Language Processing, and this project is a reflection 
        of my dedication to exploring the applications of deep learning models like BERT in the field of sentiment detection.
    """)
    
    # Add a contact form or information
    st.header("Get In Touch")
    st.write("If you'd like to reach out or collaborate, feel free to contact me at: **zahoorkaladi19@gmail.com**")

    # Interactive call-to-action buttons
    if st.button("Learn More"):
        st.write("""
            Stay tuned for more updates and features! We are constantly improving the app to make it more accurate and interactive.
        """)
    
    if st.button("Start Sentiment Analysis"):
        st.write("Navigate to the **Home** page to start analyzing Sentiments!")
