
import os
import time
from collections import Counter
from utils.css_loader import load_css
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud

# ============================
# Utility Functions
# ============================

def load_css(css_file: str):
    """
    Load a CSS file and apply it to the Streamlit app.
    """
    css_path = os.path.join(os.path.dirname(__file__), css_file)
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning(f"CSS file '{css_file}' not found. Skipping CSS loading.")

# ============================
# Model Loading
# ============================

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load the pre-trained BERT model and tokenizer from the local directory.
    This function is cached to prevent reloading on every interaction.
    """
    model_dir = os.path.join(os.path.dirname(__file__), 'model_files')
    
    try:
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        model = BertForSequenceClassification.from_pretrained(
            model_dir,
            
        )
    except Exception as e:
        st.error(f"Error loading the BERT model: {e}")
        st.stop()
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    return model, tokenizer, device

# ============================
# Dataset Definition
# ============================

class EmotionDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.texts = texts

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# ============================
# Prediction Function
# ============================

def make_predictions(input_texts, model, tokenizer, device):
    """
    Make emotion predictions on a list of input texts.
    """
    # Create dataset and dataloader
    test_dataset = EmotionDataset(input_texts, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    # Initialize LabelEncoder
    data_csv_path = os.path.join(os.path.dirname(__file__), 'data_emotion.csv')
    try:
        df = pd.read_csv(data_csv_path)
        unique_labels = df['sentiment'].unique()
        label_encoder = LabelEncoder()
        label_encoder.fit(unique_labels)
    except Exception as e:
        st.error(f"Error loading data for label encoding: {e}")
        st.stop()

    # Create label mapping (index -> label)
    label_map = {idx: label for idx, label in enumerate(label_encoder.classes_)}

    model.eval()  # Set model to evaluation mode
    all_preds = []

    # Progress bar
    progress_bar = st.progress(0)
    num_batches = len(test_dataloader)

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            # Move inputs to device
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)

            # Update progress bar
            progress_bar.progress((i + 1) / num_batches)

    # Convert predictions to labels
    predicted_labels = [label_map.get(pred, "Unknown") for pred in all_preds]
    return predicted_labels

# ============================
# Streamlit Render Function
# ============================

def render():
    """
    Render the BERT Emotion Detection page.
    """
    # Load custom CSS
    load_css("style.css")

    # HTML content for the title and description
    html_content = """
    <div class="title-container">
        <h1>
            <span>Bert Sentiment</span> <span>Detection Model</span>
        </h1>
    </div>
    <p style="text-align: center; font-size: 1.5em; color: #d5dbdb;">
        Welcome to Sentiment Detection Web App: Your Digital Companion
    </p>
    """
    st.markdown(html_content, unsafe_allow_html=True)

    # Load the model, tokenizer, and device
    model, tokenizer, device = load_model()

    # Option 1: Input text for prediction
    st.header("Input Text for Emotion Prediction")
    input_text = st.text_area("Enter text to detect emotion:")

    if st.button('Predict Emotion'):
        if input_text.strip():
            with st.spinner("Detecting emotion, please wait..."):
                # Make prediction
                result = make_predictions([input_text], model, tokenizer, device)
                st.success(f"Predicted Emotion: {result[0]}")
        else:
            st.warning("Please enter some text.")

    st.markdown("---")  # Separator

    # Option 2: Upload CSV file for predictions
    st.header("Upload CSV for Emotion Predictions")
    st.warning("This may take time depending on the size of the file, especially if the CPU is being used.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read the uploaded CSV file
            df_uploaded = pd.read_csv(uploaded_file)
            if 'content' in df_uploaded.columns:
                st.subheader("Preview of Uploaded File:")
                st.dataframe(df_uploaded.head())

                # Extract texts for prediction
                uploaded_texts = df_uploaded['content'].astype(str).tolist()

                # Make predictions
                with st.spinner("Analyzing your CSV file with BERT model... This may take a while."):
                    predicted_sentiment = make_predictions(uploaded_texts, model, tokenizer, device)

                # Add predictions to the dataframe
                df_uploaded['Predicted Emotion'] = predicted_sentiment
                st.subheader("Predicted Emotions:")
                st.dataframe(df_uploaded[['content', 'Predicted Emotion']])

                # Visualizations
                st.subheader("Visualizations")

                # Count the predicted sentiment values
                sentiment_counts = df_uploaded['Predicted Emotion'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']

                # Bar Chart
                fig_bar = px.bar(sentiment_counts, x='Sentiment', y='Count',
                                 title="Predicted Sentiment Distribution",
                                 color='Sentiment')
                st.plotly_chart(fig_bar)

                # Pie Chart
                fig_pie = px.pie(sentiment_counts, values='Count', names='Sentiment',
                                 title="Predicted Sentiment Proportions")
                st.plotly_chart(fig_pie)

                # Textual Data Visualizations
                st.subheader("Textual Data Visualizations")

                # Word Cloud for All Text Data
                st.markdown("**Word Cloud for All Text Data**")
                all_text = " ".join(df_uploaded['content'].astype(str))
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
                plt.close()

                # Word Cloud by Sentiment
                st.markdown("**Word Cloud by Sentiment**")
                sentiment_list = df_uploaded['Predicted Emotion'].unique()
                for sentiment in sentiment_list:
                    st.markdown(f"**{sentiment.capitalize()}**")
                    sentiment_text = " ".join(df_uploaded[df_uploaded['Predicted Emotion'] == sentiment]['content'].astype(str))
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(sentiment_text)
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    st.pyplot(plt)
                    plt.close()

                # Most Frequent Words by Sentiment
                st.subheader("Most Frequent Words by Sentiment")
                for sentiment in sentiment_list:
                    st.markdown(f"**{sentiment.capitalize()}**")
                    sentiment_words = " ".join(df_uploaded[df_uploaded['Predicted Emotion'] == sentiment]['content'].astype(str)).split()
                    most_common_words = Counter(sentiment_words).most_common(10)
                    common_words_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])
                    fig = px.bar(common_words_df, x='Word', y='Frequency',
                                 title=f"Top Words for {sentiment.capitalize()}")
                    st.plotly_chart(fig)

                # Basic Text Statistics
                st.subheader("Text Statistics")
                df_uploaded['word_count'] = df_uploaded['content'].astype(str).apply(lambda x: len(x.split()))
                df_uploaded['char_count'] = df_uploaded['content'].astype(str).apply(len)

                # Word Count Distribution
                st.markdown("**Word Count Distribution**")
                fig_word_count = px.histogram(df_uploaded, x='word_count', nbins=20,
                                             title="Word Count Distribution")
                st.plotly_chart(fig_word_count)

                # Character Count Distribution
                st.markdown("**Character Count Distribution**")
                fig_char_count = px.histogram(df_uploaded, x='char_count', nbins=20,
                                             title="Character Count Distribution")
                st.plotly_chart(fig_char_count)

                # Download Predictions
                csv = df_uploaded.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions", csv, file_name="predictions.csv", mime="text/csv")
            else:
                st.error("The uploaded CSV must contain a 'content' column.")
        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")

# Button for BERT Model Evaluation

# End of bert.py
# ============================
