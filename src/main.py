import json
from transformers import BertTokenizer, BertModel
from pdf_converter import convert_pdf_into_json
from preprocess import preprocess_data_pdf_to_json, load_data
from bert import train_bert_model
from displayer import generate_graph
import joblib
import os


def main(files):
    train_new_model = ""

    # Process documents if files are provided
    if files:
        train_new_model = input("Do you want to train a new model? (yes/no): ").strip().lower()

        if train_new_model == 'yes':
            list_doc = []
            list_files = []

            print("Processing files...")
            for file in files:
                list_files.append('../data/pdf/' + file + '.pdf')
                convert_pdf_into_json(file)
                processed_docs = preprocess_data_pdf_to_json(load_data('../data/raw/' + file + '.json'), file)
                list_doc.extend(processed_docs)

            print("Files processed...")

            # Convert the preprocessed data into a format compatible with the retrieve function
            # Join the words in the TaggedDocument to form the full text of the document


            print("Training model...")

            # Create the directory if it doesn't exist
            os.makedirs('../models/bert', exist_ok=True)

            # Train the BERT model and get encoded documents
            documents = [" ".join(doc.words) for doc in list_doc]
            train_bert_model(documents)

            # Save docs for future use
            with open('../models/bert/last_file.json', "w") as file_p:
                json.dump([{"words": doc.words, "tags": doc.tags} for doc in list_doc], file_p)

            print("Model trained")

        elif train_new_model == "no":

            print("Loading model...")
            if os.path.exists('../models/bert/last_file.json'):
                # Load an existing model
                with open('../models/bert/last_file.json', 'r') as file:
                    list_doc = json.load(file)
            else:
                print("This model doesn't exist, please train a new one.")
                return 1
            print("Model loaded")

    else:

        print("No file in input, going directly into loading. ")
        print("Loading model...")
        if os.path.exists('../models/bert/last_file.json'):
            # Load an existing model
            with open('../models/bert/last_file.json', 'r') as file:
                list_doc = json.load(file)
        else:
            print("This model doesn't exist, please train a new one.")
            return 1
        print("Model loaded")

    if train_new_model == "yes":
        documents = [" ".join(doc.words) for doc in list_doc]
    else:
        documents = [" ".join(doc["words"]) for doc in list_doc]

    doc_vectors = joblib.load('../models/bert/doc_vectors.pkl')
    generate_graph(doc_vectors, documents)


if __name__ == "__main__":
    main(["cognitive_neuropsycho_schizo", "Prevalence of alcohol use disorders inschizophrenia",
          "Grandiosity and Guilt Cause Paranoia"])
