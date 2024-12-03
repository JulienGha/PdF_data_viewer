import os
import re
import numpy as np
import pandas as pd
import extract_msg

# Import Counter from collections
from collections import Counter

# Text processing imports
import nltk
from nltk.corpus import stopwords

# For vectorization
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

# Clustering
import hdbscan
from sklearn.metrics import silhouette_score
import itertools

# Dimensionality reduction
import umap

# Visualization
import plotly.express as px
import plotly
import json
import matplotlib.pyplot as plt

# Flask imports
from flask import Flask, render_template, request, redirect, url_for

# Download NLTK resources if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Initialize NLTK resources
french_stop_words = stopwords.words('french')

# Initialize Flask app
app = Flask(__name__)

# Create 'static' and 'templates' directories if they don't exist
if not os.path.exists('static'):
    os.makedirs('static')
if not os.path.exists('templates'):
    os.makedirs('templates')

# Global variables to store data and cluster names
df_emails = None
cluster_names = {}
fig_json = None
X_embedded = None
cluster_keywords = {}

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

# Function to perform clustering and generate visualizations
def perform_clustering():
    global df_emails, cluster_names, fig_json, X_embedded, cluster_keywords

    # Specify the folder path
    #folder_path = r'C:\Users\JGH\Documents\Mail semaine du 4 au 8 nov'

    folder_path = r'/home/administrator/mail_infra'

    # Initialize lists to store email contents and metadata
    emails = []
    file_names = []
    email_subjects = []
    author_names = []

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

    # Create a DataFrame for easier data manipulation
    df_emails = pd.DataFrame({
        'Email': emails,
        'FileName': file_names,
        'Subject': email_subjects,
        'Author': author_names
    })

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
        n_neighbors=7,
        min_dist=1,
        metric='cosine',
        random_state=42
    )
    X_embedded = umap_reducer.fit_transform(X_np)

    # Define clustering parameters to test for HDBSCAN
    params = {
        "min_cluster_size": [3, 4, 5, 7, 10, 15, 20, 30, 50, 100],
        "min_samples": [4, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200],
        "cluster_selection_epsilon": [0.5, 0.8, 1.0, 2.0, 3.0, 5.0],
        "metric": ['euclidean']
    }

    # Initialize results list
    results = []

    # Iterate through parameter combinations
    for min_cluster_size, min_samples, epsilon, metric in itertools.product(
            params['min_cluster_size'], params['min_samples'], params['cluster_selection_epsilon'], params['metric']
    ):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_epsilon=epsilon
        )
        labels = clusterer.fit_predict(X_embedded)

        # Points in valid clusters
        non_noise_points = (labels != -1).sum()

        # Cluster coverage (proportion of non-noise points)
        cluster_coverage = non_noise_points / len(labels)

        # Number of clusters (excluding noise)
        num_clusters = len(set(labels) - {-1})

        # Calculate silhouette score for valid clusters
        if num_clusters > 1 and non_noise_points > num_clusters:
            valid_indices = labels != -1
            silhouette = silhouette_score(X_embedded[valid_indices], labels[valid_indices])
        else:
            silhouette = -1  # Invalid silhouette score

        # Define a weight for the number of clusters
        cluster_weight = 0.10  # Adjust this value to control the impact of the number of clusters

        # Composite score with a linear function of the number of clusters
        composite_score = silhouette * cluster_coverage + cluster_weight * num_clusters

        # Append results
        results.append({
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "epsilon": epsilon,
            "metric": metric,
            "noise_ratio": 1 - cluster_coverage,
            "cluster_coverage": cluster_coverage,
            "num_clusters": num_clusters,
            "silhouette_score": silhouette,
            "composite_score": composite_score
        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Sort by composite score, silhouette score, and noise ratio
    results_df = results_df.sort_values(
        by=["composite_score", "silhouette_score", "noise_ratio"],
        ascending=[False, False, True]
    )

    # Save to CSV for later analysis
    results_df.to_csv("clustering_results.csv", index=False)

    # Print the top 5 results
    print("Top 5 clustering parameter combinations:")
    print(results_df.head())

    # Select the best clustering parameters based on the results
    best_params = results_df.iloc[0]
    print("\nBest Parameters:")
    print(best_params)

    # Apply HDBSCAN clustering with the best parameters
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(best_params['min_cluster_size']),
        min_samples=int(best_params['min_samples']),
        metric=str(best_params['metric']),
        cluster_selection_epsilon=float(best_params['epsilon'])
    )
    labels = clusterer.fit_predict(X_embedded)

    # Add cluster labels to the DataFrame (keep labels as integers)
    df_emails['Cluster'] = labels  # Labels are integers

    # Initialize KeyBERT model
    kw_model = KeyBERT(model_name)

    # Extract cluster categories and topics
    cluster_texts = {}
    cluster_keywords = {}
    global cluster_names  # Use global variable
    cluster_names = {}
    cluster_indices = {}  # To store indices of emails in each cluster
    for cluster in set(labels):
        if cluster != -1:
            indices = df_emails[df_emails['Cluster'] == cluster].index  # Cluster labels are integers
            cluster_indices[cluster] = indices
            cluster_emails = df_emails.loc[indices, 'Email'].tolist()
            cluster_subjects = df_emails.loc[indices, 'Subject'].tolist()
            cluster_texts[cluster] = cluster_emails

            # Combine all cluster emails into a single document
            cluster_combined_text = ' '.join(cluster_emails)

            # Extract top keywords for the cluster
            top_keywords = kw_model.extract_keywords(
                cluster_combined_text,
                keyphrase_ngram_range=(1, 2),
                stop_words=french_stop_words,
                top_n=30
            )
            cluster_keywords[cluster] = set([kw[0] for kw in top_keywords])

            # Determine the most frequent word/phrase for naming
            word_counts = Counter()
            for phrase, score in top_keywords:
                word_counts[phrase] += 1
            most_common_word = word_counts.most_common(1)[0][0]
            cluster_names[cluster] = most_common_word.capitalize()

            # Print cluster information
            print(f"\nCluster {cluster} - {cluster_names[cluster]}:")
            print(f"Number of Emails: {len(cluster_emails)}")
            print(f"Sample Subjects: {', '.join(cluster_subjects[:5])} ...")
            print("Top Keywords:")
            for keyword, score in top_keywords[:10]:
                print(f"  - {keyword} (score: {score:.4f})")

    # Map cluster names to the DataFrame
    df_emails['Cluster_Name'] = df_emails['Cluster'].map(cluster_names)
    df_emails['Cluster_Name'] = df_emails['Cluster_Name'].fillna('Noise')

    # --------------------------
    # Reclassification (as per previous refined logic)
    # --------------------------

    # Compute centroids for each cluster
    email_embeddings = X_np
    cluster_centroids = {}
    for cluster in cluster_indices:
        indices = cluster_indices[cluster]
        cluster_embeddings = email_embeddings[indices]
        centroid = cluster_embeddings.mean(axis=0)
        cluster_centroids[cluster] = centroid

    # Precompute distances of all points to their nearest centroid
    all_distances = []
    for embedding in email_embeddings:
        min_distance = float('inf')
        for cluster, centroid in cluster_centroids.items():
            distance = np.linalg.norm(embedding - centroid)
            if distance < min_distance:
                min_distance = distance
        all_distances.append(min_distance)

    # Determine dynamic max distance threshold
    max_distance_threshold = np.percentile(all_distances, 40)  # Use the xth percentile

    print(f"Dynamic max_distance_threshold set to: {max_distance_threshold:.4f}")

    # Get indices of noise emails
    noise_indices = df_emails[df_emails['Cluster'] == -1].index  # Cluster labels are integers
    noise_embeddings = email_embeddings[noise_indices]

    # Compute density of each cluster (number of points per cluster volume)
    cluster_density = {}
    for cluster, centroid in cluster_centroids.items():
        cluster_points = email_embeddings[cluster_indices[cluster]]
        volume = np.linalg.norm(cluster_points - centroid, axis=1).sum()
        cluster_density[cluster] = len(cluster_points) / (volume + 1e-9)  # Avoid division by zero

    # Threshold for reclassification
    density_threshold = np.percentile(list(cluster_density.values()), 20)  # Use the 20th percentile

    # Assign noise points only to sufficiently dense clusters
    reclassified_clusters = []
    for idx, noise_embedding in zip(noise_indices, noise_embeddings):
        min_distance = float('inf')
        assigned_cluster = -1  # Default is noise (integer)
        for cluster, centroid in cluster_centroids.items():
            distance = np.linalg.norm(noise_embedding - centroid)
            if distance < min_distance and cluster_density[cluster] >= density_threshold:
                min_distance = distance
                closest_cluster = cluster
        if min_distance <= max_distance_threshold and cluster_density[closest_cluster] >= density_threshold:
            assigned_cluster = closest_cluster
        else:
            assigned_cluster = -1  # Remain as noise
        reclassified_clusters.append(assigned_cluster)

    # Update cluster labels for reclassified emails
    df_emails.loc[noise_indices, 'Cluster_Reclassified'] = reclassified_clusters

    # For emails that couldn't be reclassified, keep them as noise
    df_emails['Cluster_Reclassified'] = df_emails['Cluster_Reclassified'].fillna(df_emails['Cluster'])

    # Convert 'Cluster_Reclassified' to integers
    df_emails['Cluster_Reclassified'] = df_emails['Cluster_Reclassified'].astype(int)

    # Update cluster names for reclassified clusters
    df_emails['Cluster_Name_Reclassified'] = df_emails['Cluster_Reclassified'].map(cluster_names)
    df_emails['Cluster_Name_Reclassified'] = df_emails['Cluster_Name_Reclassified'].fillna('Noise')

    # --------------------------
    # Save Matplotlib Plots as Images
    # --------------------------

    # Plot the number of emails sent by each author (only those who sent more than 5 emails)
    author_counts = df_emails['Author'].value_counts()
    author_counts_filtered = author_counts[author_counts > 5]

    plt.figure(figsize=(12, 6))
    author_counts_filtered.plot(kind='bar')
    plt.title('Quantité d\'emails envoyés par personne (minimum 5)')
    plt.xlabel('Auteur')
    plt.ylabel('Quantité')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('static/emails_per_author.png')
    plt.close()

    # Group by cluster name and count the number of emails per cluster
    cluster_counts_named = df_emails.groupby(['Cluster_Reclassified', 'Cluster_Name_Reclassified']).size().reset_index(
        name='Count')

    # Calculate the total number of emails
    total_emails = cluster_counts_named['Count'].sum()

    # Identify the largest cluster
    largest_cluster = cluster_counts_named.loc[cluster_counts_named['Count'].idxmax()]

    # Check if the largest cluster exceeds 25% of the total emails
    if largest_cluster['Count'] / total_emails > 0.25:
        # Exclude the largest cluster
        cluster_counts_named_filtered = cluster_counts_named[
            cluster_counts_named['Cluster_Name_Reclassified'] != largest_cluster['Cluster_Name_Reclassified']]
        print(
            f"Excluding largest cluster: {largest_cluster['Cluster_Name_Reclassified']} with {largest_cluster['Count']} emails.")
    else:
        # Include all clusters if no cluster exceeds the threshold
        cluster_counts_named_filtered = cluster_counts_named

    # Further filter to the top 30 largest clusters
    cluster_counts_named_filtered = cluster_counts_named_filtered.nlargest(30, 'Count')

    # Plot the filtered data
    plt.figure(figsize=(10, 6))
    plt.bar(cluster_counts_named_filtered['Cluster_Name_Reclassified'], cluster_counts_named_filtered['Count'])
    plt.title('Number of Emails per Cluster (Filtered)')
    plt.xlabel('Cluster Name')
    plt.ylabel('Number of Emails')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('static/emails_per_cluster_reclassified.png')  # Save as a new file
    plt.close()

    # --------------------------
    # Save Plotly Figure as JSON
    # --------------------------

    # Prepare data for Plotly with reclassified clusters
    df_emails['Cluster_Final'] = df_emails['Cluster_Reclassified']
    df_emails['Cluster_Name_Final'] = df_emails['Cluster_Name_Reclassified']

    # Create an interactive 3D scatter plot with Plotly
    fig = px.scatter_3d(
        df_emails,
        x=X_embedded[:, 0],
        y=X_embedded[:, 1],
        z=X_embedded[:, 2],
        color='Cluster_Name_Final',
        hover_data=['Subject', 'FileName', 'Author'],
        title='Email Clusters Visualized in 3D Space After Reclassification'
    )

    fig.update_traces(marker=dict(size=5))

    # Convert Plotly figure to JSON
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# Run the clustering and visualization upon starting the app
perform_clustering()

# --------------------------
# Flask Routes and App Execution
# --------------------------

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', fig_json=fig_json)

@app.route('/authors', methods=['GET'])
def authors():
    global df_emails
    # Get the authors data
    author_counts = df_emails['Author'].value_counts().reset_index()
    author_counts.columns = ['Author', 'Email_Count']
    return render_template('authors.html', authors=author_counts.to_dict(orient='records'))

@app.route('/clusters', methods=['GET', 'POST'])
def clusters():
    global df_emails, cluster_keywords, cluster_names, fig_json, X_embedded

    if request.method == 'POST':
        # Get new names from the form
        for cluster_id in cluster_names.keys():
            new_name = request.form.get(f'cluster_{cluster_id}')
            if new_name:
                cluster_names[cluster_id] = new_name.strip()

        # Update cluster names in the DataFrame
        df_emails['Cluster_Name'] = df_emails['Cluster'].map(cluster_names)
        df_emails['Cluster_Name_Final'] = df_emails['Cluster_Name']

        # Group by cluster name and count the number of emails per cluster
        cluster_counts = df_emails['Cluster_Name_Final'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster_Name_Final', 'Count']

        # Calculate the total number of emails
        total_emails = cluster_counts['Count'].sum()

        # Identify the largest cluster
        largest_cluster_name = cluster_counts.loc[cluster_counts['Count'].idxmax(), 'Cluster_Name_Final']
        largest_cluster_count = cluster_counts['Count'].max()

        # Check if the largest cluster exceeds 25% of the total emails
        if largest_cluster_count / total_emails > 0.25:
            # Exclude the largest cluster
            cluster_counts_filtered = cluster_counts[cluster_counts['Cluster_Name_Final'] != largest_cluster_name]
            print(f"Excluding largest cluster: {largest_cluster_name} with {largest_cluster_count} emails.")
        else:
            # Include all clusters if no cluster exceeds the threshold
            cluster_counts_filtered = cluster_counts

        # Further filter to the top 30 largest clusters
        cluster_counts_filtered = cluster_counts_filtered.nlargest(30, 'Count')

        # Plot the filtered data
        plt.figure(figsize=(12, 6))
        plt.bar(cluster_counts_filtered['Cluster_Name_Final'], cluster_counts_filtered['Count'])
        plt.title('Nombre d\'Emails par Catégorie (Filtré)')
        plt.xlabel('Nom de la Catégorie')
        plt.ylabel('Nombre d\'Emails')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('static/emails_per_cluster_reclassified.png')
        plt.close()

        # Update the 3D scatter plot
        fig = px.scatter_3d(
            df_emails,
            x=X_embedded[:, 0],
            y=X_embedded[:, 1],
            z=X_embedded[:, 2],
            color='Cluster_Name_Final',
            hover_data=['Subject', 'FileName', 'Author'],
            title='Visualisation des Catégories d\'Emails en 3D (Après Renommage)'
        )
        fig.update_traces(marker=dict(size=5))
        fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return redirect(url_for('clusters'))

    # Prepare data for clusters
    cluster_info = []
    for cluster_id, cluster_name in cluster_names.items():
        count = len(df_emails[df_emails['Cluster'] == cluster_id])
        keywords = ', '.join([kw for kw in cluster_keywords.get(cluster_id, [])])  # Format keywords here
        cluster_info.append({
            'Cluster_ID': cluster_id,
            'Cluster_Name': cluster_name,
            'Email_Count': count,
            'Top_Keywords': keywords
        })
    return render_template('clusters.html', clusters=cluster_info)

@app.route('/rename_clusters', methods=['GET', 'POST'])
def rename_clusters():
    # Redirect to /clusters since we've combined the functionality
    return redirect(url_for('clusters'))

if __name__ == '__main__':
    # Ensure Flask does not reload the app multiple times
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
