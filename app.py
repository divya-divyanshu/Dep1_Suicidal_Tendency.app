import nltk
import streamlit as st
import pickle
import string
nltk.download('stopwords')
nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


st.title("Suicidal Tendency Detection System")

input_emotion = st.text_area("Enter your emotions.", "", key="input", height=100,  placeholder="Write your feelings here...")

if st.button('Predict'):
    #1 Preprocess
    transformed_emotion = transform_text(input_emotion)

    #2 Vectorization

    vector_emotion = tfidf.transform([transformed_emotion])

    #3 Predict
    result = model.predict(vector_emotion)[0]

    #4 Dispaly
    if result == 1:
        st.header("Suicidal!!!")
    else:
        st.header("You are normal")




# streamlit run app.py -- to run this file