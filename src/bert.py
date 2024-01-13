import torch
from transformers import BertTokenizer, BertModel
import umap
import joblib
import os
import numpy as np


# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_bert_model(docs):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)

    # Encode the documents using BERT
    encoded_docs = []
    for doc in docs:
        encoded_input = tokenizer(doc, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)
        encoded_docs.append(model_output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())

    encoded_docs = np.array(encoded_docs)

    # Create UMAP embeddings for the documents
    reducer = umap.UMAP(n_neighbors=45, n_components=2, min_dist=0.1, metric='cosine')
    umap_embeddings = reducer.fit_transform(encoded_docs)

    filename = '../models/bert/umap_model.sav'
    joblib.dump(reducer, filename)

    # Save the document vectors
    doc_vectors = umap_embeddings
    doc_vectors_filename = '../models/bert/doc_vectors.pkl'
    os.makedirs(os.path.dirname(doc_vectors_filename), exist_ok=True)
    joblib.dump(doc_vectors, doc_vectors_filename)

