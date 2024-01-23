import matplotlib.pyplot as plt
import numpy as np
import umap
import pandas as pd
import json
import pickle


# Function to load the BERT model's encoded documents
def load_bert_model(path="../models/bert/bert_model.pkl"):
    with open(path, "rb") as f:
        encoded_docs = pickle.load(f)
    return encoded_docs


def generate_graph():

    # create a list to store the selftext and subreddit values
    words_list = []
    tags_list = []

    with open('../models/bert/last_file.json', 'r') as file:
        documents = json.load(file)

    # iterate over the posts and extract the selftext and subreddit values
    for data in documents:
        words_list.append(data["words"])
        tags_list.append(data['tags'])

    # create a dataframe from the selftext and subreddit lists
    df = pd.DataFrame({'words': words_list, 'tags': tags_list})

    # Load the BERT model's encoded documents
    encoded_docs = load_bert_model()

    # Create UMAP embeddings for the documents
    reducer = umap.UMAP(n_neighbors=min(45, len(encoded_docs) - 1), n_components=2, min_dist=0.1, metric='cosine')
    umap_embeddings = reducer.fit_transform(encoded_docs)

    # create a new dataframe with the UMAP embeddings and subreddit column
    umap_df = pd.DataFrame(umap_embeddings, columns=['umap_1', 'umap_2'])

    umap_df['words'] = df['words']
    umap_df['tags'] = df['tags']

    # plot the UMAP embeddings with colors based on subreddit
    plt.scatter(umap_df['umap_1'], umap_df['umap_2'], s=0.05)
    plt.title('Representation of documents in 2 dimensions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

