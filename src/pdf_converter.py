import json
import csv
import PyPDF2


def convert_pdf_into_json(file):
    # Open the PDF file in input
    pdf = open('../data/pdf/' + file + '.pdf', "rb")

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
        text.append(page.extract_text())

    # Close the PDF file
    pdf.close()

    whole_text = ""

    # Split the text into lists of strings
    for i in range(num_pages):
        whole_text += text[i]

    whole_text = whole_text.replace("\n", "")
    final_text = []
    line = ""
    for char in whole_text:
        if char in [".", "!"]:
            final_text.append([line])
            line = ""
        else:
            line += char
    if line:
        final_text.append([line])

    # Create a JSON object from the list of lists of strings
    json_object = json.dumps(final_text)

    # Save the JSON object to a file
    with open('../data/raw/' + file + '.json', 'w') as f:
        f.write(json_object)


def turn_json_into_csv(file, output="default.csv"):
    # Load JSON data
    with open(file) as f:
        data = json.load(f)

    # Assuming JSON is a list of dictionaries
    keys = data[0].keys()

    # Write CSV data
    with open(output, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in data:
            writer.writerow(row)



