# ATS Resume Analyzer

This Flask-based application analyzes resumes for Applicant Tracking System (ATS) compatibility and provides a score along with feedback to help improve resume formatting and content. It supports PDF, DOCX, and TXT file uploads and utilizes natural language processing (NLP) techniques to assess resume effectiveness.

## Key Features

* **Resume Parsing:** Extracts text from PDF, DOCX, and TXT files (with OCR fallback for PDFs).
* **ATS Formatting Check:** Identifies potential formatting issues that may hinder ATS parsing (e.g., tables, columns, special characters).
* **Keyword Matching:** Extracts keywords from a job description and assesses how well the resume matches them using TF-IDF and fuzzy matching.
* **Semantic Similarity:** Calculates the semantic similarity between the resume and job description using spaCy.
* **Resume Section Identification:** Detects the presence of common resume sections (e.g., Education, Experience, Skills).
* **Comprehensive Scoring:** Provides an overall ATS score and detailed feedback on keyword matching, formatting, and content alignment.

## Technologies Used

* Python
* Flask
* PyPDF2
* docx
* NLTK
* spaCy
* scikit-learn
* pytesseract (for OCR - optional)
* pdf2image (for OCR - optional)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download the spaCy English model:**
    ```bash
    python -m spacy download en_core_web_md
    ```
5.  **(Optional) Install Tesseract OCR:**
    * If you need OCR functionality, install Tesseract OCR separately.
    * Refer to the pytesseract documentation for installation instructions: [https://github.com/madmaze/pytesseract](https://github.com/madmaze/pytesseract)
    * You may also need to install `pdf2image`.
6.  **Run the application:**
    ```bash
    python app.py
    ```

## Usage

1.  Access the application through your web browser (usually at `http://127.0.0.1:5000/`).
2.  Upload your resume file (PDF, DOCX, or TXT).
3.  Optionally, paste a job description for more targeted analysis.
4.  Click the "Analyze" button to receive the ATS score and feedback.
