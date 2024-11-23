import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import plotly.express as px
import plotly.graph_objects as go
import os
# Combined function to load model, visualize details, and display on the page
model_dir = os.path.join(os.path.dirname(__file__), 'model_files')

    
try:
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        model = BertForSequenceClassification.from_pretrained(
            model_dir,
            
        )
except Exception as e:
        st.error(f"Error loading the BERT model: {e}")
        st.stop()  
def display_model_details(model_directory):
    # Load the pre-trained model and tokenizer
    
   # tokenizer = BertTokenizer.from_pretrained(model_directory)
   # model = BertForSequenceClassification.from_pretrained(model_directory)
    
    # Display model architecture
    st.subheader("Model Architecture")
    st.text(str(model))
    
    # Display model configuration
    st.subheader("Model Configuration")
    st.json(model.config.to_dict())
    
    # Calculate the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Display parameter information
    st.subheader("Model Parameters")
    st.write(f"Total Parameters: {total_params:,}")
    st.write(f"Trainable Parameters: {trainable_params:,}")
    st.write(f"Non-Trainable Parameters: {non_trainable_params:,}")
    
    # Create interactive pie chart visualization for parameter distribution using Plotly
    param_labels = ['Trainable Parameters', 'Non-Trainable Parameters']
    param_sizes = [trainable_params, non_trainable_params]
    
    fig_pie = px.pie(
        values=param_sizes,
        names=param_labels,
        title="Model Parameters Distribution",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    
    # Display the pie chart
    st.plotly_chart(fig_pie)

    # Additional Bar chart for comparison
    fig_bar = go.Figure(data=[
        go.Bar(name='Trainable Parameters', x=['Parameters'], y=[trainable_params], marker_color='lightgreen'),
        go.Bar(name='Non-Trainable Parameters', x=['Parameters'], y=[non_trainable_params], marker_color='orange')
    ])
    fig_bar.update_layout(barmode='stack', title="Trainable vs Non-Trainable Parameters")
    
    # Display the bar chart
    st.plotly_chart(fig_bar)
    
    # Display device information
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.write(f"Model is running on: {device}")
    
    # Display tokenizer details
    st.subheader("Tokenizer Details")
    st.text(str(tokenizer))

# Function to render the page
def render():
    # Streamlit UI elements on the main page
    st.title("Pretrained BERT Model Details with Visualizations")

    st.markdown("""
    This Streamlit app displays the details of a **pretrained BERT model**. You can view:
    - Model Architecture
    - Model Configuration (Hyperparameters, Settings)
    - Model Parameters (Total, Trainable, Non-Trainable)
    - Parameter Distribution Pie Chart
    - Trainable vs Non-Trainable Parameters Bar Chart
    - Device (CPU or GPU)
    - Tokenizer Details
    """)

    # Input for the model directory
    model_directory = st.text_input("Enter the directory of the pretrained model:", "model_files")

    # Display model details if directory is provided
    if model_directory:
        display_model_details(model_directory)
