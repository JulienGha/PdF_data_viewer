import json
from pdf_converter import convert_pdf_into_json
from preprocess import preprocess_data_pdf_to_json, load_data
from bert import train_bert_model
from displayer import generate_graph_3d, extract_cluster_themes, combine_segments, generate_combined_graph_3d
import os


def main(pdf_directory, clusters_size=5, context_size=200):
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
                    convert_pdf_into_json(filename, context_size)
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

            train_bert_model(documents)

            # Save docs for future use
            with open('../models/bert/last_file.json', "w") as file_p:
                json.dump(list_doc, file_p)

            print("Model trained")
            generate_graph_3d(clusters_size)
            extract_cluster_themes()
            generate_combined_graph_3d(clusters_size)

        elif train_new_model == "no":

            print("Loading model...")
            if os.path.exists('../models/bert/last_file.json') and os.path.exists('../models/bert/bert_model.pkl'):
                generate_graph_3d(clusters_size)
                extract_cluster_themes()
                generate_combined_graph_3d(clusters_size)
        else:
            print("Need to be either yes or no!")
    else:
        print("Error, need to have pdf file in the pdf raw directory")


if __name__ == "__main__":
    pdf_directory = "../data/pdf"  # Specify the directory containing PDF files
    # Prompt the user for input until valid numbers are entered
    while True:
        try:
            cluster_size = int(input("Enter the minimum of element per cluster, lowest number means more clusters: "))
            context_size = int(input("Enter the amount of words per context, low numbers means more element on z: "))
            break  # Exit the loop if both inputs are valid numbers
        except ValueError:
            print("Please enter valid numeric values.")

    # Call the main function with user-input values
    main(pdf_directory, cluster_size, context_size)
