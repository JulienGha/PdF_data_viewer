import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import pandas as pd


def train_tfidf(docs):
    # Extract words and tags from TaggedDocument objects
    words = [doc.words for doc in docs]
    tags = [doc.tags for doc in docs]

    # create a dataframe from the selftext and subreddit lists
    df = pd.DataFrame({'text': words, 'tags': tags})

    vectorizer = TfidfVectorizer()

    # fit the vectorizer to the text and transform the data
    text_data = df['text'].apply(lambda x: ' '.join(x))
    X = vectorizer.fit_transform(text_data)

    with open('../models/tfidf/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    # Save the document vectors without using UMAP
    doc_vectors = X  # Assuming X are your document vectors
    doc_vectors_filename = '../models/tfidf/doc_vectors.pkl'
    os.makedirs(os.path.dirname(doc_vectors_filename), exist_ok=True)
    joblib.dump(doc_vectors, doc_vectors_filename)