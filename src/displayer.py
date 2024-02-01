import matplotlib.pyplot as plt
import umap
import pandas as pd
import json
import pickle
import hdbscan
from collections import Counter
from nltk.corpus import stopwords
import string
import plotly.express as px


# Function to load the BERT model's encoded documents
def load_bert_model(path="../models/bert/bert_model.pkl"):
    with open(path, "rb") as f:
        encoded_docs = pickle.load(f)
    return encoded_docs


def generate_graph_3d():
    words_list = []
    order_list = []
    doc_list = []

    with open('../models/bert/last_file.json', 'r') as file:
        documents = json.load(file)

    # iterate over the posts and extract the selftext and subreddit values
    for document in documents:
        for data in document["content"]:
            doc_list.append(document["document_name"])
            words_list.append(data["words"])
            order_list.append(data["order"])

    # create a dataframe from the selftext and subreddit lists
    df = pd.DataFrame({'words': words_list, 'order': order_list, "document": doc_list})

    # Load the BERT model's encoded documents
    encoded_docs = load_bert_model()

    # Create UMAP embeddings for the documents
    reducer = umap.UMAP(n_neighbors=min(45, len(encoded_docs) - 1), n_components=3, min_dist=0.1, metric='cosine')
    umap_embeddings = reducer.fit_transform(encoded_docs)

    # Create a new dataframe with the UMAP embeddings and subreddit column
    umap_df = pd.DataFrame(umap_embeddings, columns=['umap_1', 'umap_2', 'umap_3'])
    umap_df['words'] = df['words']
    umap_df['order'] = df['order']
    umap_df['document'] = df['document']

    # Use HDBSCAN to cluster the UMAP embeddings (considering only umap_1 and umap_2)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=1)
    umap_df['cluster'] = clusterer.fit_predict(umap_df[['umap_1', 'umap_2']])

    # Save the data to a CSV file ordered by clusters
    csv_path = "../models/bert/umap_clusters.csv"
    umap_df_sorted = umap_df.sort_values(by='cluster')  # Sort by cluster
    umap_df_sorted.to_csv(csv_path, index=False)

    # Plot the UMAP embeddings with colors based on clusters and lines for the same document
    fig = px.scatter_3d(umap_df, x='x', y='y', z='order', color='cluster', size_max=10,
                        hover_data=['document', 'order'])
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(title='Representation of documents in 3D with HDBSCAN clusters')
    fig.show()


generate_graph_3d()



def extract_cluster_themes():

    # Load the BERT model's encoded documents
    encoded_docs = load_bert_model()

    # Use HDBSCAN to cluster the UMAP embeddings
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=1)
    clusters = clusterer.fit_predict(encoded_docs)

    # Load the original documents
    with open('../models/bert/last_file.json', 'r') as file:
        documents = json.load(file)

    # Create a DataFrame with document text and assigned clusters
    df = pd.DataFrame({'text': [data["words"] for data in documents], 'cluster': clusters})

    # Extract themes for each cluster excluding stop words and punctuation
    cluster_themes = {}
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)

    for cluster_id in df['cluster'].unique():
        cluster_texts = df[df['cluster'] == cluster_id]['text']
        flattened_texts = [word.lower() for sublist in cluster_texts for word in sublist
                           if word.lower() not in stop_words and word not in punctuation and len(word) > 3]
        cluster_word_counts = Counter(flattened_texts)
        most_common_words = cluster_word_counts.most_common(30)
        cluster_themes[cluster_id] = most_common_words

    # Print or use cluster themes as needed
    for cluster_id, theme in cluster_themes.items():
        print(f"Cluster {cluster_id} Theme: {theme}")
