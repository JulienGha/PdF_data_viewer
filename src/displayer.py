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
import nltk
from bert import train_bert_model

nltk.download('stopwords')


# Function to load the BERT model's encoded documents
def load_bert_model(path="../models/bert/bert_model.pkl"):
    with open(path, "rb") as f:
        encoded_docs = pickle.load(f)
    return encoded_docs


def generate_graph_3d(size_clusters, combined=False):
    words_list = []
    order_list = []
    doc_list = []

    with open('../models/bert/last_file.json', 'r') as file:
        documents = json.load(file)

    for document in documents:
        for data in document["content"]:
            doc_list.append(document["document_name"])
            words_list.append(data["words"])
            order_list.append(data["order"])

    # Create a dataframe from the lists
    df = pd.DataFrame({'words': words_list, 'order': order_list, "document": doc_list})

    # Load the BERT model's encoded documents
    encoded_docs = load_bert_model()

    # Create UMAP embeddings for the documents
    reducer = umap.UMAP(n_neighbors=min(15, len(encoded_docs) - 1), n_components=2, min_dist=0.05, metric='cosine')
    umap_embeddings = reducer.fit_transform(encoded_docs)

    # Create a new dataframe with the UMAP embeddings and document columns
    umap_df = pd.DataFrame(umap_embeddings, columns=['umap_1', 'umap_2'])
    umap_df['words'] = df['words']
    umap_df['order'] = df['order']
    umap_df['document'] = df['document']

    # Use HDBSCAN to cluster the UMAP embeddings
    clusterer = hdbscan.HDBSCAN(min_cluster_size=size_clusters, min_samples=1)
    umap_df['cluster'] = clusterer.fit_predict(umap_df[['umap_1', 'umap_2']])

    # Save the data to a CSV file ordered by clusters
    csv_path = "../models/bert/umap_clusters.csv"
    umap_df_sorted = umap_df.sort_values(by=['document', 'order'])  # Sort by document and order
    umap_df_sorted.to_csv(csv_path, index=False)

    # Plot the UMAP embeddings with colors based on clusters and lines for the same document
    fig = px.scatter_3d(umap_df, x='umap_1', y='umap_2', z='order', color='cluster', size_max=10,
                        hover_data=['document', 'order'])

    # Add lines connecting points within each document
    for document in umap_df['document'].unique():
        document_data = umap_df[umap_df['document'] == document].sort_values(by='order')
        fig.add_trace(px.line_3d(document_data, x='umap_1', y='umap_2', z='order').data[0])

    fig.update_traces(marker=dict(size=4))
    title_suffix = " (Combined Segments)" if combined else ""
    fig.update_layout(title=f'Representation of documents in 3D with HDBSCAN clusters{title_suffix}')
    fig.show()

    return umap_df


def extract_cluster_themes():
    # Load data from the CSV file
    csv_path = "../models/bert/umap_clusters.csv"
    umap_df = pd.read_csv(csv_path)

    # Extract themes for each cluster excluding stop words and punctuation
    cluster_themes = {}
    stop_words = set(stopwords.words('french'))
    punctuation = set(string.punctuation)

    for cluster_id in umap_df['cluster'].unique():
        cluster_texts = umap_df[umap_df['cluster'] == cluster_id]['words']
        flattened_texts = [word.lower() for sublist in cluster_texts for word in str(sublist).split()
                           if word.lower() not in stop_words and word not in punctuation and len(word) > 3]
        cluster_word_counts = Counter(flattened_texts)
        most_common_words = cluster_word_counts.most_common(30)
        cluster_themes[int(cluster_id)] = most_common_words  # Convert cluster_id to int

    # Print or use cluster themes as needed
    for cluster_id, theme in cluster_themes.items():
        print(f"Cluster {cluster_id} Theme: {theme}")


def combine_segments():
    csv_path = "../models/bert/umap_clusters.csv"
    umap_df = pd.read_csv(csv_path)

    combined_segments = []

    # Group by document and cluster, and sort by order within each group
    grouped = umap_df.groupby(['document', 'cluster'])

    for (document, cluster), group in grouped:
        group_sorted = group.sort_values(by='order')
        current_text = ""
        previous_order = None

        for _, row in group_sorted.iterrows():
            current_order = row['order']
            text = row['words']

            # Ensure text is a string and handle NaNs
            if isinstance(text, float) and pd.isna(text):
                continue
            text = str(text)

            # Combine if contiguous
            if previous_order is None or current_order == previous_order + 1:
                current_text += " " + text
            else:
                combined_segments.append({
                    "document": document,
                    "cluster": cluster,
                    "order": previous_order,  # Use the previous order to indicate the end of the segment
                    "words": current_text.strip()
                })
                current_text = text

            previous_order = current_order

        # Add the last segment
        if current_text:
            combined_segments.append({
                "document": document,
                "cluster": cluster,
                "order": previous_order,
                "words": current_text.strip()
            })

    combined_segments_df = pd.DataFrame(combined_segments)
    combined_segments_path = "../models/bert/combined_segments.csv"
    combined_segments_df.to_csv(combined_segments_path, index=False)

    print(f"Combined segments saved to {combined_segments_path}")
    return combined_segments_df


def generate_combined_graph_3d(size_clusters):
    combined_segments_df = combine_segments()

    # Encode combined segments using the BERT model
    combined_docs = combined_segments_df['words'].tolist()
    encoded_combined_docs = train_bert_model(combined_docs)

    # Create UMAP embeddings for the combined documents
    reducer = umap.UMAP(n_neighbors=min(15, len(encoded_combined_docs) - 1), n_components=2, min_dist=0.05,
                        metric='cosine')
    umap_embeddings_combined = reducer.fit_transform(encoded_combined_docs)

    # Create a new dataframe with the UMAP embeddings and cluster column
    umap_df_combined = pd.DataFrame(umap_embeddings_combined, columns=['umap_1', 'umap_2'])
    umap_df_combined['words'] = combined_segments_df['words']
    umap_df_combined['order'] = combined_segments_df['order']
    umap_df_combined['document'] = combined_segments_df['document']
    umap_df_combined['cluster'] = combined_segments_df['cluster']

    # Plot the UMAP embeddings with colors based on clusters
    fig = px.scatter_3d(umap_df_combined, x='umap_1', y='umap_2', z='order', color='cluster', size_max=10,
                        hover_data=['document', 'order', 'words'])

    fig.update_traces(marker=dict(size=4))
    fig.update_layout(title='Representation of Combined Segments in 3D with HDBSCAN clusters')
    fig.show()
