import os
import re
import numpy as np
import pandas as pd
import extract_msg

# Text processing imports
import nltk
from nltk.corpus import stopwords

# For vectorization
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

# Clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Dimensionality reduction
import umap

# Visualization
import plotly.express as px
import plotly
import json

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
    global df_emails, cluster_names, fig_json, X_embedded

    # Specify the folder path
    folder_path = r'/home/administrator/mail_infra'  # Update this path to your folder

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

    # Dimensionality reduction using UMAP to 3D for visualization
    umap_reducer = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    X_embedded = umap_reducer.fit_transform(X_np)

    # --------------------------
    # K-Means Clustering
    # --------------------------

    # Determine the optimal number of clusters using the Elbow Method
    # You can adjust this range as needed
    cluster_range = range(5, 21)  # Trying 5 to 20 clusters
    inertia = []
    silhouette_scores = []

    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_np)
        inertia.append(kmeans.inertia_)
        silhouette = silhouette_score(X_np, labels)
        silhouette_scores.append(silhouette)
        print(f"Clusters: {n_clusters}, Inertia: {kmeans.inertia_:.2f}, Silhouette Score: {silhouette:.4f}")

    # Choose the number of clusters with the highest silhouette score
    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_clusters}")

    # Apply K-Means with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    labels = kmeans.fit_predict(X_np)

    # Add cluster labels to the DataFrame
    df_emails['Cluster'] = labels

    # Initialize KeyBERT model
    kw_model = KeyBERT(model_name)

    # Extract cluster categories and topics
    cluster_names = {}
    for cluster in set(labels):
        indices = df_emails[df_emails['Cluster'] == cluster].index
        cluster_emails = df_emails.loc[indices, 'Email'].tolist()
        cluster_subjects = df_emails.loc[indices, 'Subject'].tolist()

        # Combine all cluster emails into a single document
        cluster_combined_text = ' '.join(cluster_emails)

        # Extract top keywords for the cluster
        top_keywords = kw_model.extract_keywords(
            cluster_combined_text,
            keyphrase_ngram_range=(1, 2),
            stop_words=french_stop_words,
            top_n=20
        )
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

    # Prepare data for Plotly with clusters
    df_emails['Cluster_Final'] = df_emails['Cluster']
    df_emails['Cluster_Name_Final'] = df_emails['Cluster_Name']

    # --------------------------
    # Save Plotly Figure as JSON
    # --------------------------

    # Create an interactive 3D scatter plot with Plotly
    fig = px.scatter_3d(
        df_emails,
        x=X_embedded[:, 0],
        y=X_embedded[:, 1],
        z=X_embedded[:, 2],
        color='Cluster_Name_Final',
        hover_data=['Subject', 'FileName', 'Author'],
        title='Email Clusters Visualized in 3D Space'
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

@app.route('/rename_clusters', methods=['GET', 'POST'])
def rename_clusters():
    global cluster_names, df_emails, fig_json, X_embedded

    if request.method == 'POST':
        # Get new names from the form
        for cluster_id in cluster_names.keys():
            new_name = request.form.get(f'cluster_{cluster_id}')
            if new_name:
                cluster_names[cluster_id] = new_name.strip()

        # Update cluster names in the DataFrame
        df_emails['Cluster_Name'] = df_emails['Cluster'].map(cluster_names)
        df_emails['Cluster_Name'] = df_emails['Cluster_Name'].fillna('Noise')

        df_emails['Cluster_Name_Final'] = df_emails['Cluster_Name']

        # Update the 3D scatter plot
        fig = px.scatter_3d(
            df_emails,
            x=X_embedded[:, 0],
            y=X_embedded[:, 1],
            z=X_embedded[:, 2],
            color='Cluster_Name_Final',
            hover_data=['Subject', 'FileName', 'Author'],
            title='Email Clusters Visualized in 3D Space After Renaming'
        )
        fig.update_traces(marker=dict(size=5))
        fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return redirect(url_for('index'))

    # For GET request, render the renaming form
    return render_template('rename_clusters.html', cluster_names=cluster_names)

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
