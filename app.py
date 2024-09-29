import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize the PorterStemmer
ps = PorterStemmer()

from sklearn.feature_extraction.text import TfidfVectorizer

# Assuming `train_sms` contains the training data
tfidf = TfidfVectorizer()

# Save the fitted vectorizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess and transform text
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the text

    y = []
    for i in text:
        if i.isalnum():  # Keep only alphanumeric tokens
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # Stem the words

    return " ".join(y)

# Load pre-trained vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit interface for the app
st.title("Email/SMS Spam Classifier")

# Text area for user input
input_sms = st.text_area("Enter the message")

# Predict button
if st.button('Predict'):

    # 1. Preprocess the input message
    transformed_sms = transform_text(input_sms)
    
    # 2. Vectorize the preprocessed message
    vector_input = tfidf.transform([transformed_sms])
    
    # 3. Predict whether it's spam or not
    result = model.predict(vector_input)[0]
    
    # 4. Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
