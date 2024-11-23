import nltk
import string
from nltk.corpus import stopwords

nltk.download('stopwords')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = ''.join([char.lower() for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)
