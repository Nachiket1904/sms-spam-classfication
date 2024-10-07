# Import necessary libraries
import nltk
nltk.download('stopwords')  # Download stopwords from NLTK
nltk.download('punkt')  # Download the Punkt tokenizer from NLTK

import streamlit as st  # Streamlit for web app development
import pickle  # To load the saved model and vectorizer
import string  # For handling punctuation
from nltk.corpus import stopwords  # Stopwords to remove common English words
from nltk.stem.porter import PorterStemmer  # PorterStemmer for stemming words

# Initialize the PorterStemmer
ps = PorterStemmer()

# Function to preprocess the input text
def transform_text(text):
    # 1. Convert text to lowercase
    text = text.lower()

    # 2. Tokenize the text (split into individual words)
    text = nltk.word_tokenize(text)

    # 3. Remove non-alphanumeric characters (keep only words and numbers)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # 4. Remove stopwords (common words) and punctuation
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # 5. Apply stemming (reduce words to their root form)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    # 6. Return the preprocessed text as a single string
    return " ".join(y)

# Load the TF-IDF vectorizer and the pre-trained classification model
tfidf = pickle.load(open('vectorizer1.pkl', 'rb'))
model = pickle.load(open('model1.pkl', 'rb'))

# Streamlit app title
st.title("Email/SMS Spam Classifier")

# Text input box for the user to enter the message
input_sms = st.text_area("Enter the message")

# Button to trigger the prediction
if st.button('Predict'):
    
    # 1. Preprocess the input message
    transformed_sms = transform_text(input_sms)
    
    # 2. Vectorize the preprocessed message using the TF-IDF vectorizer
    vector_input = tfidf.transform([transformed_sms])
    
    # 3. Use the loaded model to make a prediction (Spam or Not Spam)
    result = model.predict(vector_input)[0]
    
    # 4. Display the result on the app
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
