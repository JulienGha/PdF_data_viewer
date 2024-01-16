import matplotlib.pyplot as plt
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

    vectorizer = TfidfVectorizer(stop_words=stop_words)

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

    # create a dictionary to map subreddit names to colors
    subreddit_color_dict = {
        'BPD': 'red',
        'Anxiety': 'blue',
        'bipolar': 'green',
        'depression': 'pink',
        'schizophrenia': 'yellow',
        'mentalillness': 'orange',
        'others': 'purple'
        # add more subreddits and corresponding colors as needed
    }

    # create a list of colors based on the subreddit column
    colors = [subreddit_color_dict[subreddit] for subreddit in umap_df['subreddit']]


    # plot the UMAP embeddings with colors based on subreddit
    plt.scatter(umap_df['umap_1'], umap_df['umap_2'], c=colors, s=0.05)
    plt.title('UMAP embeddings with subreddit colors')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()