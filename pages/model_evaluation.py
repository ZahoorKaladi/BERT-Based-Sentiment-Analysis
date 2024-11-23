import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize

def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig)

def plot_classification_report(y_test, y_pred, labels):
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    fig = px.bar(report_df[:-3].reset_index(), 
                 x='index', y=['precision', 'recall', 'f1-score'],
                 barmode='group', title="Classification Report Metrics")
    st.plotly_chart(fig)

def render():
    st.header("Model Evaluation")

    if st.button("Evaluate Model"):
        y_pred = st.session_state.model.predict(st.session_state.X_test)

        # Accuracy score
        accuracy = accuracy_score(st.session_state.y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2%}")

        # Classification report
        st.write("Classification Report:")
        plot_classification_report(st.session_state.y_test, y_pred, st.session_state.model.classes_)

        # Confusion matrix
        cm = confusion_matrix(st.session_state.y_test, y_pred)
        st.write("Confusion Matrix:")
        plot_confusion_matrix(cm, st.session_state.model.classes_)
