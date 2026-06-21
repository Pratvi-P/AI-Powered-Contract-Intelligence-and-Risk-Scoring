from pdf2image import convert_from_path
import pytesseract
import os


# Tesseract location
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# Poppler location
POPPLER_PATH = (
    r"C:\Users\DELL\Downloads\Release-26.02.0-0"
    r"\poppler-26.02.0\Library\bin"
)


def extract_text_from_pdf(pdf_path):
    """
    Convert PDF pages to images and extract text using OCR.
    """

    print("Converting PDF pages into images...")

    pages = convert_from_path(
        pdf_path,
        poppler_path=POPPLER_PATH
    )

    print(f"Total pages found: {len(pages)}")

    full_text = ""

    for index, page in enumerate(pages, start=1):
        print(f"Processing page {index}")

        text = pytesseract.image_to_string(
            page,
            config="--oem 3 --psm 6"
        )

        full_text += text + "\n"

    return full_text


def save_text(text, filename):
    """
    Save extracted text into a file.
    """

    os.makedirs("extracted_data", exist_ok=True)

    path = os.path.join("extracted_data", filename)

    with open(path, "w", encoding="utf-8") as file:
        file.write(text)

    print(f"Text saved successfully: {path}")


if __name__ == "__main__":

    pdf_file = "data/contracts/sample.pdf"

    print("Starting OCR pipeline...\n")

    extracted_text = extract_text_from_pdf(pdf_file)

    print("\n========== EXTRACTED TEXT ==========\n")
    print(extracted_text[:1000])

    save_text(
        extracted_text,
        "contract_001.txt"
    )

    print("\nOCR pipeline completed successfully!")