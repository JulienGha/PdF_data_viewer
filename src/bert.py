from transformers import BertTokenizer, BertModel
import torch
import pickle

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to save the BERT model's encoded documents
def save_bert_model(encoded_docs, path="../models/bert/bert_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(encoded_docs, f)


# Function to train a BERT model (for encoding documents)
def train_bert_model(documents):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)

    # Encoding the documents
    encoded_docs = []
    for doc in documents:
        encoded_input = tokenizer(doc, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)
        encoded_docs.append(model_output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
    save_bert_model(encoded_docs)
    return encoded_docs

