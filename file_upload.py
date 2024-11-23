import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def upload_and_predict():
    st.header("Upload Your File")
    
    # File uploader for CSV files
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        # Use spinner to show loading while the file is being processed and analyzed
        with st.spinner('Processing file and analyzing sentiments, please wait...'):
            # Read the uploaded CSV file
            unseen_data = pd.read_csv(uploaded_file)
            
            # Ensure the file contains a 'content' column
            if 'content' in unseen_data.columns:
                # Preprocess the text data in the 'content' column
                unseen_data['processed_content'] = unseen_data['content'].apply(preprocess_text)
                
                # Transform the preprocessed content using the vectorizer
                vectorized_unseen = st.session_state.vectorizer.transform(unseen_data['processed_content'])
                
                # Predict sentiments using the trained model
                unseen_data['predicted_sentiment'] = st.session_state.model.predict(vectorized_unseen)
                
                # Show success message
                st.success('Emotion analysis completed!')

                # Display the original content and predicted sentiment
                st.write(unseen_data[['content', 'predicted_sentiment']])
                
                # Count the predicted sentiment values
                sentiment_counts = unseen_data['predicted_sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                
                # Plot bar chart of predicted sentiment distribution
                fig_bar = px.bar(sentiment_counts, x='Sentiment', y='Count', 
                                 title="Predicted Sentiment Distribution", color='Sentiment')
                st.plotly_chart(fig_bar)

                # Plot pie chart of predicted sentiment proportions
                fig_pie = px.pie(sentiment_counts, values='Count', names='Sentiment', 
                                 title="Predicted Sentiment Proportions")
                st.plotly_chart(fig_pie)
                
                # Textual data visualizations
                st.subheader("Textual Data Visualizations")

                # Word cloud of the entire text data
                st.markdown("**Word Cloud for All Text Data**")
                all_text = " ".join(unseen_data['processed_content'])
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)

                # Word cloud by sentiment
                st.markdown("**Word Cloud by Sentiment**")
                sentiment_list = unseen_data['predicted_sentiment'].unique()
                for sentiment in sentiment_list:
                    st.markdown(f"**{sentiment.capitalize()}**")
                    sentiment_text = " ".join(unseen_data[unseen_data['predicted_sentiment'] == sentiment]['processed_content'])
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(sentiment_text)
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    st.pyplot(plt)
                
                # Most frequent words by sentiment category
                st.subheader("Most Frequent Words by Sentiment")
                for sentiment in sentiment_list:
                    st.markdown(f"**{sentiment.capitalize()}**")
                    sentiment_words = " ".join(unseen_data[unseen_data['predicted_sentiment'] == sentiment]['processed_content']).split()
                    most_common_words = Counter(sentiment_words).most_common(10)
                    common_words_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])
                    fig = px.bar(common_words_df, x='Word', y='Frequency', title=f"Top Words for {sentiment.capitalize()}")
                    st.plotly_chart(fig)
                
                # Basic text statistics
                st.subheader("Text Statistics")
                unseen_data['word_count'] = unseen_data['processed_content'].apply(lambda x: len(x.split()))
                unseen_data['char_count'] = unseen_data['processed_content'].apply(lambda x: len(x))

                st.markdown("**Word Count Distribution**")
                fig_word_count = px.histogram(unseen_data, x='word_count', nbins=20, title="Word Count Distribution")
                st.plotly_chart(fig_word_count)

                st.markdown("**Character Count Distribution**")
                fig_char_count = px.histogram(unseen_data, x='char_count', nbins=20, title="Character Count Distribution")
                st.plotly_chart(fig_char_count)
                
            else:
                # Show error if the required 'content' column is missing
                st.error("The uploaded file must contain a 'content' column.")

# Assuming a text preprocessing function exists
def preprocess_text(text):
    # Implement your preprocessing logic here (e.g., lowercasing, removing stopwords, etc.)
    return text
