import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import io
from sklearn.metrics import classification_report, accuracy_score
import time
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

# Function to load the model with a loading spinner

# Function to load the model with a loading spinner
@st.cache_resource
def load_model():
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.vectorizer = None
        st.session_state.X_test = None
        st.session_state.y_test = None

    if st.session_state.model is None:
        st.header("Model Training")
        with st.spinner("Training the model, please wait..."):
            time.sleep(3)
      # Add delay to simulate model loading time

        model_directory = "model_files"
        
        # Load the tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_directory)
        # Simulate loading time for tokenizer
       # time.sleep(1)

        # Load the model
        model = BertForSequenceClassification.from_pretrained(model_directory)
        
        # Simulate loading time for model
        #time.sleep(2)

        # Check if CUDA (GPU) is available and set the model to the appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
    st.success(f"Model loaded successfully on {device}!")
    return model, tokenizer, device

# Load the model at the beginning of the app
model, tokenizer, device = load_model()

# Define the EmotionDataset class
class EmotionDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])



# Function to make predictions
# Function to make predictions
def make_predictions(input_texts):
    # Create dataset and dataloader for new texts
    test_dataset = EmotionDataset(input_texts)
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    # Check if CUDA is available and move the model to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Initialize the LabelEncoder with the same labels used during training
    df = pd.read_csv('data_emotion.csv')  # Replace with your actual training dataset
    unique_labels = df['sentiment']
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_labels)

    # Create the label mapping (index -> label)
    label_map = {idx: label for idx, label in enumerate(label_encoder.classes_)}

    model.eval()  # Set model to evaluation mode
    all_preds = []

    # Update progress bar during prediction
    with torch.no_grad():  # Disable gradient calculations for evaluation
        num_batches = len(test_dataloader)
        progress_bar = st.progress(0)  # Initialize progress bar
        for i, batch in enumerate(test_dataloader):
            # Move input tensors to the device
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            outputs = model(**inputs)  # Get the model outputs
            logits = outputs.logits  # Extract the logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()  # Get the predicted labels
            all_preds.extend(preds)

            # Update the progress bar correctly
            progress_bar.progress(min((i + 1) / num_batches, 1.0))  # Ensure it reaches 100%

    # Convert predictions to human-readable labels using the label mapping
    predicted_labels = [label_map.get(pred, "Unknown") for pred in all_preds]

    return predicted_labels

# Streamlit app layout
st.title("Emotion Detection via BERT")

# Option 1: Input text for prediction
st.write("Input Text for Emotion Prediction")
input_text = st.text_area("Enter text to detect emotion:")

if st.button('Predict Emotion'):
    if input_text:
        with st.spinner("Detecting , please wait..."):
            time.sleep(2)   
        result = make_predictions([input_text])
        st.write("Predicted Emotion:", result[0])
    else:
        st.warning("Please enter some text.")

# Option 2: Upload CSV file for predictions
st.header("Upload CSV for Emotion Predictions")
st.warning("This may take time depending on the size of the file, especially if the CPU is being used.")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    df_uploaded = pd.read_csv(uploaded_file)
    if 'content' in df_uploaded.columns:
        st.write("Preview of uploaded file:")
        st.write(df_uploaded.head())

        # Make predictions on the uploaded CSV
        uploaded_texts = df_uploaded['content'].tolist()

        # Display spinner while processing the CSV file
        with st.spinner("Analyzing your CSV file with BERT model... This may take a while."):
            predicted_sentiment = make_predictions(uploaded_texts)

        # Add predictions to the dataframe and display
        df_uploaded['Predicted Emotion'] = predicted_sentiment
        st.write("Predicted Emotions:")
        st.write(df_uploaded[['content', 'Predicted Emotion']])

        # Count the predicted sentiment values
        sentiment_counts = df_uploaded['Predicted Emotion'].value_counts().reset_index()
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
        all_text = " ".join(df_uploaded['content'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)

        # Word cloud by sentiment
        st.markdown("**Word Cloud by Sentiment**")
        sentiment_list = df_uploaded['Predicted Emotion'].unique()
        for sentiment in sentiment_list:
            st.markdown(f"**{sentiment.capitalize()}**")
            sentiment_text = " ".join(df_uploaded[df_uploaded['Predicted Emotion'] == sentiment]['content'])
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(sentiment_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)

        # Most frequent words by sentiment category
        st.subheader("Most Frequent Words by Sentiment")
        for sentiment in sentiment_list:
            st.markdown(f"**{sentiment.capitalize()}**")
            sentiment_words = " ".join(df_uploaded[df_uploaded['Predicted Emotion'] == sentiment]['content']).split()
            most_common_words = Counter(sentiment_words).most_common(10)
            common_words_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])
            fig = px.bar(common_words_df, x='Word', y='Frequency', title=f"Top Words for {sentiment.capitalize()}")
            st.plotly_chart(fig)

        # Basic text statistics
        st.subheader("Text Statistics")
        df_uploaded['word_count'] = df_uploaded['content'].apply(lambda x: len(x.split()))
        df_uploaded['char_count'] = df_uploaded['content'].apply(lambda x: len(x))

        st.markdown("**Word Count Distribution**")
        fig_word_count = px.histogram(df_uploaded, x='word_count', nbins=20, title="Word Count Distribution")
        st.plotly_chart(fig_word_count)

        st.markdown("**Character Count Distribution**")
        fig_char_count = px.histogram(df_uploaded, x='char_count', nbins=20, title="Character Count Distribution")
        st.plotly_chart(fig_char_count)

        # Optionally, allow the user to download the results as a CSV
        csv = df_uploaded.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv, file_name="predictions.csv", mime="text/csv")
    else:
        st.error("The uploaded CSV must contain a 'content' column.")
