import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import time

from utils.preprocess import preprocess_text

def train_model():
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.vectorizer = None
        st.session_state.X_test = None
        st.session_state.y_test = None

    if st.session_state.model is None:
        st.header("Model Training")
        with st.spinner("Training the model, please wait..."):
            time.sleep(2)

            dataset = pd.read_csv("data_emotion.csv")
            df = pd.DataFrame(dataset)
            df['processed_content'] = df['content'].apply(preprocess_text)

            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(df['processed_content'])
            X_train, X_test, y_train, y_test = train_test_split(X, df['sentiment'], test_size=0.2, random_state=42)

            model = LogisticRegression()
            model.fit(X_train, y_train)

            st.session_state.model = model
            st.session_state.vectorizer = vectorizer
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

            st.success("Model successfully trained.")
    else:
        st.success("Model loaded from memory.")
