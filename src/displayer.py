import matplotlib.pyplot as plt
import numpy as np
import umap
import pandas as pd


def generate_graph(vectors, documents):

    # create a list to store the selftext and subreddit values
    selftext_list = []
    subreddit_list = []

    # iterate over the posts and extract the selftext and subreddit values
    for data in documents:
        selftext_list.append()
        subreddit_list.append(data['subreddit'])


    # create a dataframe from the selftext and subreddit lists
    df = pd.DataFrame({'selftext': selftext_list, 'subreddit': subreddit_list})

    encoded_docs = np.array(encoded_docs)

    # Create UMAP embeddings for the documents
    reducer = umap.UMAP(n_neighbors=45, n_components=2, min_dist=0.1, metric='cosine')
    umap_embeddings = reducer.fit_transform(encoded_docs)

    # fit the vectorizer to the post text and transform the data
    text_data = df['selftext']
    X = vectorizer.fit_transform(text_data)

    # create UMAP embeddings for the posts
    umap_embeddings = umap.UMAP(n_neighbors=45,
                                n_components=2,  # reduce to 2 dimensions for visualization
                                min_dist=0.1,
                                metric='cosine').fit_transform(X)

    # create a new dataframe with the UMAP embeddings and subreddit column
    umap_df = pd.DataFrame(umap_embeddings, columns=['umap_1', 'umap_2'])

    umap_df['subreddit'] = df['subreddit']
    umap_df['selftext'] = df['selftext']

    # plot the UMAP embeddings with colors based on subreddit
    plt.scatter(umap_df['umap_1'], umap_df['umap_2'], s=0.05)
    plt.title('Representation of documents in 2 dimension')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
