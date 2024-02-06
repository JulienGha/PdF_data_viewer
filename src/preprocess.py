import json


def preprocess_data_pdf_to_json(documents, document_name="default"):
    data = {
        "document_name": document_name,
        "content": []
    }
    for i, entry in enumerate(documents):
        if len(entry) > 50:
            subdocument = {"words": entry, "order": i}
            data["content"].append(subdocument)

    return data


def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
