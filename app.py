import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize



ps = PorterStemmer()


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


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email / SMS Spam Detection")

input_sms = st.text_input("Enter Your Massage")


if st.button("Predict"):
    # 1. Preprocess
    transform_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transform_sms])

    # 3. Prediction
    prediction = model.predict(vector_input)

    # 4. Display
    if prediction == 1:
        st.success("Your Massage was spam")

    else:
        st.warning("Your Massage was not spam")