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
    folder_path = r'/home/administrator/mail_infra'  # Update this path as needed
    #folder_path = r'C:\Users\JGH\Documents\Mail semaine du 4 au 8 nov'  # Update this path as needed

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
        min_dist=0.6,
        metric='cosine',
        random_state=42
    )
    X_embedded = umap_reducer.fit_transform(X_np)

    # Initial clustering parameters
    params = {
        "min_cluster_size": [3, 4, 5, 7, 10],
        "min_samples": [1, 2, 3, 4],
        "cluster_selection_epsilon": [0.5, 1.0, 2.0],
        "metric": ['euclidean']
    }

    # Initialize results list
    results = []
    clusterer_configs = []

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

        # Unique labels and number of clusters
        unique_labels = set(labels)
        num_clusters = len(unique_labels - {-1})

        # Compute statistics
        noise_points = (labels == -1).sum()
        noise_ratio = noise_points / len(labels)

        # Calculate cluster sizes (excluding noise)
        cluster_sizes = [sum(labels == cluster) for cluster in unique_labels - {-1}]
        if cluster_sizes:
            average_cluster_size = sum(cluster_sizes) / num_clusters
            size_difference = max(cluster_sizes) - min(cluster_sizes)
        else:
            average_cluster_size = 0
            size_difference = 0

        # Append results
        results.append({
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "epsilon": epsilon,
            "num_clusters": num_clusters,
            "noise_ratio": noise_ratio,
            "average_cluster_size": average_cluster_size,
            "size_difference": size_difference
        })
        clusterer_configs.append(clusterer)

        # Print clustering results for debugging
        print(f"\nParameters: min_cluster_size={min_cluster_size}, "
              f"min_samples={min_samples}, epsilon={epsilon}")
        print(f"Unique labels: {unique_labels}")
        print(f"Number of clusters found: {num_clusters}")
        print(f"Noise ratio: {noise_ratio:.2f}")
        print(f"Average cluster size: {average_cluster_size}")
        print(f"Size difference: {size_difference}")

    # Configure pandas to display all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    # Display options in console
    results_df = pd.DataFrame(results)
    print("\nAvailable Clustering Configurations:")
    print(results_df[[
        "min_cluster_size",
        "min_samples",
        "epsilon",
        "num_clusters",
        "noise_ratio",
        "average_cluster_size",
        "size_difference"
    ]])

    # User selects the desired configuration
    choice = int(input("\nSelect the configuration index (0-based): "))
    best_params = results_df.iloc[choice]
    print(f"\nUsing Parameters: {best_params.to_dict()}")

    # Apply the chosen configuration
    chosen_clusterer = clusterer_configs[choice]
    labels = chosen_clusterer.fit_predict(X_embedded)
    df_emails['Cluster'] = labels

    # Initialize KeyBERT model
    kw_model = KeyBERT(model_name)

    # Initialize global variables
    global cluster_names
    cluster_names = {}
    cluster_indices = {}
    cluster_keywords = {}

    # Function to extract cluster names and keywords
    def extract_cluster_info(labels):
        unique_clusters = set(labels) - {-1}
        for cluster in unique_clusters:
            indices = df_emails[df_emails['Cluster'] == cluster].index
            cluster_indices[cluster] = indices
            cluster_emails = df_emails.loc[indices, 'Email'].tolist()
            cluster_subjects = df_emails.loc[indices, 'Subject'].tolist()

            # Combine all cluster emails into a single document
            cluster_combined_text = ' '.join(cluster_emails)

            # Extract top keywords for the cluster
            top_keywords = kw_model.extract_keywords(
                cluster_combined_text,
                keyphrase_ngram_range=(1, 3),
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

    # Extract initial cluster information
    extract_cluster_info(labels)

    # --------------------------
    # Iterative Reclustering
    # --------------------------
    max_iterations = 100  # Maximum number of reclustering iterations
    iteration = 0
    total_emails = len(df_emails)

    while True:
        # Identify the largest cluster
        cluster_counts = df_emails['Cluster'].value_counts()
        largest_cluster_label = cluster_counts.idxmax()
        largest_cluster_size = cluster_counts.max()
        size_difference = cluster_counts.max() - cluster_counts.min()

        print(f"\nIteration {iteration + 1}:")
        print(f"Largest cluster {largest_cluster_label} has {largest_cluster_size} emails "
              f"({largest_cluster_size / total_emails:.2%} of total).")
        print(f"Size difference between largest and smallest clusters: {size_difference}")

        # Check if the largest cluster contains fewer than 200 emails or maximum iterations reached
        if largest_cluster_size <= 600 or iteration >= max_iterations:
            print("Stopping reclustering as conditions are met.")
            break

        # Proceed to recluster the largest cluster
        print(f"Reclustering cluster {largest_cluster_label}...")

        # Extract data points belonging to the largest cluster
        indices_largest_cluster = df_emails[df_emails['Cluster'] == largest_cluster_label].index
        X_largest_cluster = X_np[indices_largest_cluster]

        # Apply UMAP again for this subset (optional)
        umap_reducer_sub = umap.UMAP(
            n_components=3,
            n_neighbors=7,
            min_dist=0.6,
            metric='cosine',
            random_state=42
        )
        X_embedded_sub = umap_reducer_sub.fit_transform(X_largest_cluster)

        # Define new clustering parameters for reclustering
        recluster_params = {
            "min_cluster_size": [3, 4, 5, 7, 10],
            "min_samples": [1, 2, 3, 4],
            "cluster_selection_epsilon": [0.5, 1.0, 2.0],
            "metric": ['euclidean']
        }

        # Initialize reclustering results
        recluster_results = []
        reclusterer_configs = []

        # Iterate through reclustering parameter combinations
        for min_cluster_size, min_samples, epsilon, metric in itertools.product(
                recluster_params['min_cluster_size'], recluster_params['min_samples'],
                recluster_params['cluster_selection_epsilon'], recluster_params['metric']
        ):
            reclusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                cluster_selection_epsilon=epsilon
            )
            sub_labels = reclusterer.fit_predict(X_embedded_sub)

            # Unique labels and number of clusters
            unique_sub_labels = set(sub_labels)
            num_sub_clusters = len(unique_sub_labels - {-1})

            # Compute statistics
            sub_cluster_sizes = [sum(sub_labels == cluster) for cluster in unique_sub_labels - {-1}]
            if sub_cluster_sizes:
                sub_average_cluster_size = sum(sub_cluster_sizes) / num_sub_clusters
                sub_size_difference = max(sub_cluster_sizes) - min(sub_cluster_sizes)
            else:
                sub_average_cluster_size = 0
                sub_size_difference = 0

            # Append results
            recluster_results.append({
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "epsilon": epsilon,
                "num_clusters": num_sub_clusters,
                "average_cluster_size": sub_average_cluster_size,
                "size_difference": sub_size_difference
            })
            reclusterer_configs.append(reclusterer)

        # Create a DataFrame to store results
        recluster_results_df = pd.DataFrame(recluster_results)

        # Define a composite score for each configuration
        recluster_results_df['composite_score'] = (
                recluster_results_df['num_clusters'] * 2  # Higher weight for number of clusters
                - recluster_results_df['average_cluster_size']  # Penalize larger average cluster sizes
        )

        # Select the reclustering configuration with the highest composite score
        best_recluster_index = recluster_results_df['composite_score'].idxmax()
        best_recluster_params = recluster_results_df.iloc[best_recluster_index]
        print(f"Best reclustering parameters: {best_recluster_params.to_dict()}")

        # Apply the best reclustering configuration
        best_reclusterer = reclusterer_configs[best_recluster_index]
        sub_labels = best_reclusterer.fit_predict(X_embedded_sub)

        # Offset subcluster labels to ensure unique labels
        max_label = df_emails['Cluster'].max()
        sub_labels_adjusted = sub_labels.copy()
        sub_labels_adjusted[sub_labels != -1] += max_label + 1

        # Update labels in the original DataFrame
        df_emails.loc[indices_largest_cluster, 'Cluster'] = sub_labels_adjusted

        # Update cluster names and keywords for new subclusters
        new_clusters = set(sub_labels_adjusted) - {-1}
        for cluster in new_clusters:
            indices = df_emails[df_emails['Cluster'] == cluster].index
            cluster_emails = df_emails.loc[indices, 'Email'].tolist()
            cluster_subjects = df_emails.loc[indices, 'Subject'].tolist()

            # Combine all cluster emails into a single document
            cluster_combined_text = ' '.join(cluster_emails)

            # Extract top keywords for the cluster
            top_keywords = kw_model.extract_keywords(
                cluster_combined_text,
                keyphrase_ngram_range=(1, 4),
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

            # Print reclustered cluster information
            print(f"\nReclustered Cluster {cluster} - {cluster_names[cluster]}:")
            print(f"Number of Emails: {len(cluster_emails)}")
            print(f"Sample Subjects: {', '.join(cluster_subjects[:5])} ...")
            print("Top Keywords:")
            for keyword, score in top_keywords[:10]:
                print(f"  - {keyword} (score: {score:.4f})")

        # Update cluster names in the DataFrame
        df_emails['Cluster_Name'] = df_emails['Cluster'].map(cluster_names)
        df_emails['Cluster_Name'] = df_emails['Cluster_Name'].fillna('Noise')

        iteration += 1

    # --------------------------
    # Reclassification Step
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

    # Remove NaN values from all_distances
    all_distances = [d for d in all_distances if not np.isnan(d)]

    if all_distances:
        max_distance_threshold = np.percentile(all_distances, 10)
        print(f"Dynamic max_distance_threshold set to: {max_distance_threshold:.4f}")
    else:
        print("No valid distances found. Setting max_distance_threshold to default value.")
        max_distance_threshold = 0

    # Get indices of noise emails
    noise_indices = df_emails[df_emails['Cluster'] == -1].index
    noise_embeddings = email_embeddings[noise_indices]

    # Compute density of each cluster (number of points per cluster volume)
    cluster_density = {}
    for cluster, centroid in cluster_centroids.items():
        cluster_points = email_embeddings[cluster_indices[cluster]]
        if len(cluster_points) > 0:
            volume = np.linalg.norm(cluster_points - centroid, axis=1).sum()
            cluster_density[cluster] = len(cluster_points) / (volume + 1e-9)  # Avoid division by zero
        else:
            print(f"Cluster {cluster} has no points.")

    # Handle empty cluster_density
    if cluster_density:
        density_threshold = np.percentile(list(cluster_density.values()), 10)
    else:
        print("No clusters to compute density threshold.")
        density_threshold = 0

    # Assign noise points only to sufficiently dense clusters
    reclassified_clusters = []
    for idx, noise_embedding in zip(noise_indices, noise_embeddings):
        min_distance = float('inf')
        assigned_cluster = -1  # Default is noise
        closest_cluster = -1
        for cluster, centroid in cluster_centroids.items():
            distance = np.linalg.norm(noise_embedding - centroid)
            if distance < min_distance and cluster_density[cluster] >= density_threshold:
                min_distance = distance
                closest_cluster = cluster
        if min_distance <= max_distance_threshold and cluster_density.get(closest_cluster, 0) >= density_threshold:
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
    plt.title("Quantité d'emails envoyés par personne (minimum 5)")
    plt.xlabel('Auteur')
    plt.ylabel('Quantité')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('static/emails_per_author.png')
    plt.close()

    # Group by cluster name and count the number of emails per cluster
    cluster_counts_named = df_emails.groupby(
        ['Cluster_Reclassified', 'Cluster_Name_Reclassified']
    ).size().reset_index(name='Count')

    # Calculate the total number of emails
    total_emails = cluster_counts_named['Count'].sum()

    # Identify the largest cluster
    if not cluster_counts_named.empty:
        largest_cluster = cluster_counts_named.loc[cluster_counts_named['Count'].idxmax()]
        # Check if the largest cluster exceeds 25% of the total emails
        if largest_cluster['Count'] / total_emails > 0.25:
            # Exclude the largest cluster
            cluster_counts_named_filtered = cluster_counts_named[
                cluster_counts_named['Cluster_Name_Reclassified'] != largest_cluster['Cluster_Name_Reclassified']
            ]
            print(
                f"Excluding largest cluster: {largest_cluster['Cluster_Name_Reclassified']} "
                f"with {largest_cluster['Count']} emails."
            )
        else:
            # Include all clusters if no cluster exceeds the threshold
            cluster_counts_named_filtered = cluster_counts_named
    else:
        cluster_counts_named_filtered = cluster_counts_named

    # Further filter to the top 30 largest clusters
    cluster_counts_named_filtered = cluster_counts_named_filtered.nlargest(30, 'Count')

    # Plot the filtered data
    plt.figure(figsize=(10, 6))
    plt.bar(
        cluster_counts_named_filtered['Cluster_Name_Reclassified'],
        cluster_counts_named_filtered['Count']
    )
    plt.title('Number of Emails per Cluster (Filtered)')
    plt.xlabel('Cluster Name')
    plt.ylabel('Number of Emails')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('static/emails_per_cluster_reclassified.png')
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
        title='Email Clusters Visualized in 3D Space After Iterative Reclustering'
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