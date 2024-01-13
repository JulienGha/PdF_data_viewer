import json
from pdf_converter import convert_pdf_into_json
from preprocess import preprocess_data_pdf_to_json, load_data
from doc2vec import train_doc2vec
from bert import train_bert_model
from tfidf import train_tfidf
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
            model_choice = ""

            while model_choice not in ["doc", "bert", "tfidf"]:
                model_choice = input("Do you want doc2vec, BERT or tfidf? (doc/bert/tfidf): ").strip().lower()

                if model_choice == "doc":
                    print("Training model...")

                    # Create the directory if it doesn't exist
                    os.makedirs('../models/doc2vec', exist_ok=True)

                    # Train the Doc2Vec model
                    train_doc2vec(list_doc)

                    # Save docs for future use
                    with open('../models/doc2vec/last_file.json', "w") as file_p:
                        json.dump([{"words": doc.words, "tags": doc.tags} for doc in list_doc], file_p)

                    print("Model trained")

                elif model_choice == "bert":
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

                elif model_choice == "tfidf":
                    print("Training model...")

                    # Create the directory if it doesn't exist
                    os.makedirs('../models/tfidf', exist_ok=True)

                    train_tfidf(list_doc)

                    # Save the model for future use
                    with open('../models/tfidf/last_file.json', "w") as file_p:
                        json.dump([{"words": doc.words, "tags": doc.tags} for doc in list_doc], file_p)

                    print("Model trained")

        elif train_new_model == "no":
            model_choice = ""

            while model_choice not in ["doc", "bert", "tfidf"]:
                model_choice = input(
                    "Do you want to load doc2vec, BERT or tfidf? (doc/bert/tfidf): ").strip().lower()

                if model_choice == "doc":
                    print("Loading model...")
                    if os.path.exists('../models/doc2vec/last_file.json'):
                        # Load an existing model
                        with open('../models/doc2vec/last_file.json', 'r') as file:
                            list_doc = json.load(file)
                    else:
                        print("This model doesn't exist, please train a new one.")
                        break
                    print("Model loaded")

                elif model_choice == "bert":
                    print("Loading model...")
                    if os.path.exists('../models/bert/last_file.json'):
                        # Load an existing model
                        with open('../models/bert/last_file.json', 'r') as file:
                            list_doc = json.load(file)
                    else:
                        print("This model doesn't exist, please train a new one.")
                        break
                    print("Model loaded")

                elif model_choice == "tfidf":
                    if os.path.exists('../models/tfidf/last_file.json'):
                        # Load an existing model
                        with open('../models/tfidf/last_file.json', 'r') as file:
                            list_doc = json.load(file)
                    else:
                        print("This model doesn't exist, please train a new one.")
                        break
                    print("Model loaded")

    else:
        model_choice = ""

        while model_choice not in ["doc", "bert", "tfidf"]:
            model_choice = input("No file in input, going directly into loading. "
                                 "Do you want to load doc2vec, BERT or tfidf? (doc/bert/tfidf): ").strip().lower()

            if model_choice == "doc":
                print("Loading model...")
                if os.path.exists('../models/doc2vec/last_file.json'):
                    # Load an existing model
                    with open('../models/doc2vec/last_file.json', 'r') as file:
                        list_doc = json.load(file)
                else:
                    print("This model doesn't exist, please train a new one.")
                    break
                print("Model loaded")

            elif model_choice == "bert":
                print("Loading model...")
                if os.path.exists('../models/bert/last_file.json'):
                    # Load an existing model
                    with open('../models/bert/last_file.json', 'r') as file:
                        list_doc = json.load(file)
                else:
                    print("This model doesn't exist, please train a new one.")
                    break
                print("Model loaded")

            elif model_choice == "tfidf":
                print("Loading Model...")
                if os.path.exists('../models/tfidf/last_file.json'):
                    # Load an existing model
                    with open('../models/tfidf/last_file.json', 'r') as file:
                        list_doc = json.load(file)
                else:
                    print("This model doesn't exist, please train a new one.")
                    break
                print("Model loaded")

    if train_new_model == "yes":
        documents = [" ".join(doc.words) for doc in list_doc]
    else:
        documents = [" ".join(doc["words"]) for doc in list_doc]

    # User input for retrieval
    while True:
        query = input("Enter your query (or 'exit' to quit): ").strip()

        if query.lower() == 'exit':
            break


        if model_choice == "doc":
            print()

        elif model_choice == "bert":
            print()

        elif model_choice == "tfidf":
            print()



if __name__ == "__main__":
    main(["cognitive_neuropsycho_schizo", "Prevalence of alcohol use disorders inschizophrenia",
          "Grandiosity and Guilt Cause Paranoia"])
