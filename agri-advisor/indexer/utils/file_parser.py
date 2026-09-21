import PyPDF2
import os

class PDFParser:
    """
    PDFParser provides utilities for reading PDF files and extracting their textual content.
    It supports extracting all text from a PDF and saving the extracted text to a file.
    This class is useful for converting PDF documents into plain text for further processing or analysis.
    """

    def __init__(self, file_path):
        """
        Initialize the PDFParser with the path to the PDF file.

        :param file_path: Path to the PDF file to be parsed.
        """
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")

    def extract_text(self):
        """
        Extracts and returns all text from the PDF file.

        :return: Extracted text as a string.
        """
        pdf_reader = PyPDF2.PdfReader(self.file_path)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text

    def save_text_to_file(self, output_path):
        """
        Extracts text from the PDF and saves it to a text file.

        :param output_path: Path to the output text file.
        """
        text = self.extract_text()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

if __name__ == "__main__":
    # Use absolute path or ensure the PDF file is in the correct location
    pdf_path = os.path.join(os.path.dirname(__file__), "AR_Eng_2024_25.pdf")
    output_txt_path = os.path.join(os.path.dirname(__file__), "AR_Eng_2024_25.txt")
    try:
        parser = PDFParser(pdf_path)
        parser.save_text_to_file(output_txt_path)
        print(f"Extracted text saved to: {output_txt_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure 'AR_Eng_2024_25.pdf' exists in the same directory as this script.")
    except Exception as e:
        print(f"Error parsing PDF: {e}")