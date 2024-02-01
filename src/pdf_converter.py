import json
import PyPDF2


def convert_pdf_into_json(file):
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
        text.append(content.replace("\n", ""))

    # Close the PDF file
    pdf.close()

    # Create a JSON object from the list of lists of strings
    json_object = json.dumps(text)

    # Save the JSON object to a file
    with open('../data/raw/' + file.replace(".pdf", "") + '.json', 'w') as f:
        f.write(json_object)
