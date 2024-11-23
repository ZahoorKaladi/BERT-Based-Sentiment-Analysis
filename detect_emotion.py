import streamlit as st 
import nltk
from utils.preprocess import preprocess_text
import time
nltk.download('stopwords')
def detect_emotion():
    st.header("Sentiment Analysis ")
    input_text = st.text_area("Enter text to detect emotion:", key="detect_text_area")

    if st.button("Detect Sentiment", key="detect_button"):
        if input_text:
         with st.spinner("Detecting , please wait..."):
            time.sleep(2)    # Preprocess and predict emotion
            preprocessed_text = preprocess_text(input_text)
            vectorized_text = st.session_state.vectorizer.transform([preprocessed_text])
            predicted_emotion = st.session_state.model.predict(vectorized_text)[0]
            st.write(f"Detected Sentiments: **{predicted_emotion}**")
        else:
            st.warning("Please enter some text.")