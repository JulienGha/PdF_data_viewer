import os
import re
import extract_msg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
import umap
import hdbscan
import numpy as np
import nltk
from collections import defaultdict

# Visualization imports
import plotly.express as px
import pandas as pd

# Download NLTK resources if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords

# Specify the folder path
folder_path = r'C:\Users\JGH\Documents\Mail semaine du 4 au 8 nov'

# Initialize lists to store email contents and filenames
emails = []
file_names = []
email_subjects = []
author_names = []

# Function to clean and preprocess text
def preprocess_text(text, author_name):
    # Remove special characters and digits
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()

    # Remove common email phrases
    common_phrases = ['cordialement', 'bien à vous', 'merci d avance', 'bonjour', 'salutations',
                      'christophe de almeida', 'david erard', 'lestoises', 'ch', 'www', 'mailto',
                      'jonathan ponti']
    for phrase in common_phrases:
        text = text.replace(phrase, '')

    # Tokenize and remove stop words
    tokens = nltk.word_tokenize(text)
    french_stop_words = stopwords.words('french')

    # Add the author's name to the stop words
    if author_name:
        author_tokens = nltk.word_tokenize(author_name.lower())
        french_stop_words.extend(author_tokens)

    # Remove stop words and author name tokens
    tokens = [token for token in tokens if token not in french_stop_words]

    # Return the processed text without stemming
    return ' '.join(tokens)

# Iterate over all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.msg'):
        file_path = os.path.join(folder_path, file_name)
        try:
            # Extract email content using extract_msg
            msg = extract_msg.Message(file_path)
            msg_sender = msg.sender if msg.sender else ''
            msg_subject = msg.subject if msg.subject else ''
            msg_body = msg.body if msg.body else ''
            full_text = msg_subject + ' ' + msg_body

            # Preprocess the email text
            processed_text = preprocess_text(full_text, msg_sender)

            emails.append(processed_text)
            file_names.append(file_name)
            email_subjects.append(msg_subject)
            author_names.append(msg_sender)
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

# Filter out short emails
min_length = 0  # Minimum number of tokens
filtered_emails = []
filtered_file_names = []
filtered_subjects = []
filtered_authors = []

for email, file_name, subject, author in zip(emails, file_names, email_subjects, author_names):
    if len(email.split()) >= min_length:
        filtered_emails.append(email)
        filtered_file_names.append(file_name)
        filtered_subjects.append(subject)
        filtered_authors.append(author)

emails = filtered_emails
file_names = filtered_file_names
email_subjects = filtered_subjects
author_names = filtered_authors

# Convert the emails into TF-IDF vectors
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),  # Unigrams and bigrams
    max_df=0.95,
    min_df=2
)
X = vectorizer.fit_transform(emails)

# Normalize the TF-IDF vectors
normalizer = Normalizer()
X_normalized = normalizer.fit_transform(X)

# Dimensionality reduction using UMAP to 3D
umap_reducer = umap.UMAP(
    n_components=3,
    n_neighbors=5,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)
X_embedded = umap_reducer.fit_transform(X_normalized)

# Apply HDBSCAN clustering
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=4,
    min_samples=10,
    metric='euclidean',
    cluster_selection_epsilon=0.1
)
cluster_labels = clusterer.fit_predict(X_embedded)

# Output the cluster labels for each email
for file_name, label in zip(file_names, cluster_labels):
    print(f'Email: {file_name}, Cluster: {label}')



# Extract cluster categories
cluster_categories = defaultdict(list)

for email, cluster in zip(emails, cluster_labels):
    if cluster != -1:  # Ignore noise
        cluster_categories[cluster].append(email)

# Analyze representative keywords for each cluster
def get_top_keywords(tfidf_matrix, cluster_indices, feature_names, top_n=10):
    """
    Extract top n keywords for each cluster using their TF-IDF scores.
    """
    # Get the rows corresponding to the cluster
    cluster_matrix = tfidf_matrix[cluster_indices]

    # Compute the mean TF-IDF score for each feature in the cluster
    cluster_tfidf = cluster_matrix.mean(axis=0)

    # Convert to a 1D array
    cluster_tfidf_array = np.asarray(cluster_tfidf).flatten()

    # Get the indices of the top n features
    sorted_indices = np.argsort(-cluster_tfidf_array)[:top_n]
    return [feature_names[i] for i in sorted_indices]

feature_names = vectorizer.get_feature_names_out()
category_keywords = {}

for cluster in cluster_categories:
    cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
    top_keywords = get_top_keywords(X, cluster_indices, feature_names)
    category_keywords[cluster] = top_keywords

# Output the cluster labels and keywords
print("\nCluster Categories:")
for cluster, keywords in category_keywords.items():
    print(f"Cluster {cluster}: {', '.join(keywords)}")

# Prepare data for Plotly
df = pd.DataFrame({
    'UMAP1': X_embedded[:, 0],
    'UMAP2': X_embedded[:, 1],
    'UMAP3': X_embedded[:, 2],
    'Cluster': cluster_labels.astype(str),
    'Subject': email_subjects,
    'FileName': file_names,
    'Author': author_names
})

# Create an interactive 3D scatter plot with Plotly
fig = px.scatter_3d(
    df,
    x='UMAP1',
    y='UMAP2',
    z='UMAP3',
    color='Cluster',
    hover_data=['Subject', 'FileName', 'Author'],
    title='Email Clusters Visualized in 3D Space'
)

fig.update_traces(marker=dict(size=5))
fig.show()
