import streamlit as st
from pages import home, model_evaluation, twitter_integration, web_scraping, about
from bert_model import bert, Model_Evoluation

# Page selection
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Bert Model", "Bert Model Evaluation","ML Model","ML Model Evaluation", "Twitter API Integration", "Web Scraping", "About Us"])

# Render selected page
if page == "ML Model":
    home.render()
elif page == "Bert Model":
    bert.render()  # This will load the BERT model interface
elif page == "ML Model Evaluation":
    model_evaluation.render()
elif page == "Bert Model Evaluation":
    Model_Evoluation.render()
elif page == "Twitter API Integration":
    twitter_integration.render()
elif page == "Web Scraping":
    web_scraping.render()
elif page == "About Us":
    about.render()
