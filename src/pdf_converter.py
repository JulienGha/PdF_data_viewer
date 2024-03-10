import json
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import string

nltk.download('punkt')
nltk.download('stopwords')


def convert_pdf_into_json(file, context=100):
    # Open the PDF file in input
    pdf = open('../data/pdf/' + file, "rb")

    # Create a PDF reader object
    reader = PyPDF2.PdfReader(pdf)

    # Get the number of pages in the PDF file
    num_pages = len(reader.pages)

    # Create an empty list to store the text from the PDF file
    text = []

    # Iterate over the pages in the PDF file
    for i in range(num_pages):
        # Get the text from the current page
        page = reader.pages[i]
        content = page.extract_text()

        # Split the content into sentences
        sentences = sent_tokenize(content)

        # Process each sentence
        current_sentence = ""
        for sentence in sentences:
            words_in_sentence = [word for word in word_tokenize(sentence) if word.isalnum()]

            # Check if adding the current sentence exceeds the context size
            if len(current_sentence.split()) + len(words_in_sentence) <= context:
                current_sentence += " ".join(words_in_sentence)
            else:
                # Add the current sentence to the list and reset for the next one
                text.append(current_sentence.strip())
                current_sentence = " ".join(words_in_sentence)

            # Add punctuation to the end of the current sentence
            current_sentence += sentence[-1] if sentence[-1] in string.punctuation else ""

        # If there's any remaining content in the current sentence, add it to the list
        if current_sentence.strip():
            text.append(current_sentence.strip())

    # Close the PDF file
    pdf.close()

    # Create a JSON object from the list of lists of strings
    json_object = json.dumps(text)

    # Save the JSON object to a file
    with open('../data/raw/' + file.replace(".pdf", "") + '.json', 'w') as f:
        f.write(json_object)
