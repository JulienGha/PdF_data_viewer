import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict
import itertools
import extract_msg

# Text processing imports
import nltk
from nltk.corpus import stopwords

# For vectorization
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

# Dimensionality reduction and clustering
import umap
from sklearn.cluster import DBSCAN

# Visualization
import plotly.express as px

# Metrics
from sklearn.metrics import silhouette_score

# Download NLTK resources if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Initialize NLTK resources
french_stop_words = stopwords.words('french')

# Specify the folder path
folder_path = r'C:\Users\JGH\Documents\Mail semaine du 4 au 8 nov'  # Update this path to your folder

# Initialize lists to store email contents and metadata
emails = []
file_names = []
email_subjects = []
author_names = []

# Function to clean and preprocess text
def preprocess_text(text, author_name):
    # Remove email addresses and URLs
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'http\S+', ' ', text)

    # Remove special characters and digits
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.lower()

    # Remove common email phrases
    common_phrases = [
        ' cordialement ', ' bien à vous ', ' merci d avance ', ' bonjour ', 'meilleures salutations ',
        ' salutations ', 'yannick technicien', ' christophe de almeida', 'david erard', ' lestoises ', ' ch ', 'www', ' mailto ',
        ' jonathan ponti ', ' psychiatrie ', ' psychothérapie ', ' lausanne ', ' toises ', ' fax ', ' centre ',
        ' tel ', ' tél ', ' mousquines ', ' avenue '
    ]
    for phrase in common_phrases:
        text = text.replace(phrase, ' ')

    # Tokenize and remove stop words
    tokens = nltk.word_tokenize(text)

    # Add the author's name to the stop words
    if author_name:
        author_tokens = nltk.word_tokenize(author_name.lower())
        french_stop_words.extend(author_tokens)

    # Remove stop words and author name tokens
    tokens = [token for token in tokens if token not in french_stop_words]

    # Return the processed text
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
min_length = 10  # Minimum number of tokens
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

# Vectorize emails using a multilingual model
model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(model_name)

# Encode the emails to get embeddings
X = model.encode(emails, convert_to_tensor=True)

# Convert embeddings to NumPy array
X_np = X.detach().cpu().numpy()

# Dimensionality reduction using UMAP to 3D
umap_reducer = umap.UMAP(
    n_components=3,
    n_neighbors=10,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)
X_embedded = umap_reducer.fit_transform(X_np)

# Define clustering parameters to test for DBSCAN
params = {
    "eps": [0.2, 0.5, 0.7, 0.9, 1.0],
    "min_samples": [3, 5, 7, 10],
    "metric": ['euclidean', 'cosine']
}

# Initialize results list
results = []

# Iterate through parameter combinations
for eps, min_samples, metric in itertools.product(
        params['eps'], params['min_samples'], params['metric']
):
    clusterer = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric
    )
    labels = clusterer.fit_predict(X_embedded)

    # Calculate noise ratio
    noise_ratio = (labels == -1).sum() / len(labels)

    # Calculate number of clusters (excluding noise)
    num_clusters = len(np.unique(labels[labels != -1]))

    # Calculate silhouette score for valid clusters
    if num_clusters > 1 and len(labels[labels != -1]) > num_clusters:
        valid_indices = labels != -1
        silhouette = silhouette_score(X_embedded[valid_indices], labels[valid_indices])
    else:
        silhouette = -1  # Invalid silhouette score

    # Append results
    results.append({
        "eps": eps,
        "min_samples": min_samples,
        "metric": metric,
        "noise_ratio": noise_ratio,
        "num_clusters": num_clusters,
        "silhouette_score": silhouette
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Filter out invalid entries
valid_results_df = results_df[results_df['silhouette_score'] > 0]

if valid_results_df.empty:
    print("No valid parameter combinations found.")
    # Handle this case appropriately, e.g., adjust parameters or exit
    exit()
else:
    # Sort by silhouette score and noise ratio
    valid_results_df = valid_results_df.sort_values(by=["silhouette_score", "noise_ratio"], ascending=[False, True])

    # Save to CSV for later analysis
    valid_results_df.to_csv("clustering_results_dbscan.csv", index=False)

    # Print the top 5 results
    print("Top 5 clustering parameter combinations:")
    print(valid_results_df.head())

    # Select the best clustering parameters based on the results
    # Here, we take the parameters from the first row of the sorted DataFrame
    best_params = valid_results_df.iloc[0]

    # Print the best parameters and their types
    print("\nBest Parameters:")
    print(best_params)
    print("\nParameter Types:")
    print(f"eps type: {type(best_params['eps'])}")
    print(f"min_samples type: {type(best_params['min_samples'])}")
    print(f"metric type: {type(best_params['metric'])}")

    # Apply DBSCAN clustering with the best parameters
    clusterer = DBSCAN(
        eps=float(best_params['eps']),
        min_samples=int(5),
        metric=str(best_params['metric'])
    )
    labels = clusterer.fit_predict(X_embedded)

    # Check if valid clusters are found
    if len(set(labels)) <= 1:
        print("No valid clusters found with the best parameters.")
        # Decide whether to proceed, adjust parameters, or exit
        exit()

    # Initialize KeyBERT model
    kw_model = KeyBERT(model_name)

    # Extract cluster categories and topics
    cluster_texts = {}
    for cluster in set(labels):
        if cluster != -1:
            indices = [i for i, label in enumerate(labels) if label == cluster]
            cluster_emails = [emails[i] for i in indices]
            cluster_subjects = [email_subjects[i] for i in indices]
            cluster_texts[cluster] = cluster_emails

            # Combine all cluster emails into a single document
            cluster_combined_text = ' '.join(cluster_emails)

            # Extract top keywords for the cluster
            top_keywords = kw_model.extract_keywords(
                cluster_combined_text,
                keyphrase_ngram_range=(1, 2),
                stop_words=french_stop_words,
                top_n=10
            )

            # Print cluster information
            print(f"\nCluster {cluster}:")
            print(f"Number of Emails: {len(cluster_emails)}")
            print(f"Sample Subjects: {', '.join(cluster_subjects[:5])} ...")  # Show first 5 subjects
            print("Top Keywords:")
            for keyword, score in top_keywords:
                print(f"  - {keyword} (score: {score:.4f})")

    # Prepare data for Plotly
    df = pd.DataFrame({
        'UMAP1': X_embedded[:, 0],
        'UMAP2': X_embedded[:, 1],
        'UMAP3': X_embedded[:, 2],
        'Cluster': labels.astype(str),
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
        title='Email Clusters Visualized in 3D Space (DBSCAN)'
    )

    fig.update_traces(marker=dict(size=5))
    fig.show()
