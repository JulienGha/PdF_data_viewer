import json
from pdf_converter import convert_pdf_into_json
from preprocess import preprocess_data_pdf_to_json, load_data
from bert import train_bert_model
from displayer import generate_graph, extract_cluster_themes
import os


def main(pdf_directory):
    train_new_model = ""
    if not os.path.exists('../data/raw'):
        os.makedirs('../data/raw')
    if not os.path.exists('../data/pdf'):
        os.makedirs('../data/pdf')
    if not os.path.exists('../models/bert'):
        os.makedirs('../models/bert')

    # Process documents in the specified directory
    if pdf_directory:
        train_new_model = input("Do you want to train a new model? (yes/no): ").strip().lower()

        if train_new_model == 'yes':
            list_doc = []
            list_files = []

            print("Processing files...")
            for filename in os.listdir(pdf_directory):
                if filename.endswith(".pdf"):
                    file_path = os.path.join(pdf_directory, filename)
                    list_files.append(file_path)
                    convert_pdf_into_json(filename)
                    filename = filename.replace(".pdf", "")
                    processed_docs = preprocess_data_pdf_to_json(load_data('../data/raw/' + filename + '.json'),
                                                                 filename)
                    list_doc.append(processed_docs)

            print("Files processed...")

            # Convert the preprocessed data into a format compatible with the retrieve function
            # Join the words in the TaggedDocument to form the full text of the document

            print("Training model...")

            # Train the BERT model and get encoded documents
            documents = [(subdoc["words"]) for doc in list_doc for subdoc in doc["content"]]

            print(documents)

            """train_bert_model(documents)

            # Save docs for future use
            with open('../models/bert/last_file.json', "w") as file_p:
                json.dump(list_doc, file_p)

            print("Model trained")
            generate_graph()
            extract_cluster_themes()"""

        elif train_new_model == "no":

            print("Loading model...")
            if os.path.exists('../models/bert/last_file.json') and os.path.exists('../models/bert/bert_model.pkl'):
                generate_graph()
                extract_cluster_themes()


if __name__ == "__main__":
    pdf_directory = "../data/pdf"  # Specify the directory containing PDF files
    main(pdf_directory)
