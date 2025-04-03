from flask import Flask, request, jsonify, render_template
import os
import traceback
from werkzeug.utils import secure_filename
import logging
import re
import subprocess
import json
import difflib
# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'txt'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Import dependencies with proper error handling
try:
    import PyPDF2
    import docx
    import pandas as pd
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

    # spaCy initialization with lemmatization
    import spacy
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        logger.info("Downloading spaCy model...")
        subprocess.check_call(["python", "-m", "spacy", "download", "en_core_web_md"])
        nlp = spacy.load("en_core_web_md")

    # Import OCR tools with graceful degradation
    try:
        import pytesseract
        from pdf2image import convert_from_path
        OCR_AVAILABLE = True
    except ImportError:
        logger.warning("OCR tools not available. Some PDF processing may be limited.")
        OCR_AVAILABLE = False

except ImportError as e:
    logger.error(f"Missing required dependency: {str(e)}")
    raise

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyPDF2 with OCR fallback."""
    logger.debug(f"Extracting text from PDF: {pdf_path}")
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text() or ""
                text += page_text
    except Exception as e:
        logger.error(f"Error reading PDF with PyPDF2: {str(e)}")

    # If no text, try OCR
    if not text.strip() and OCR_AVAILABLE:
        try:
            logger.info("No text extracted via PyPDF2. Falling back to OCR...")
            images = convert_from_path(pdf_path)
            ocr_text = ""
            for image in images:
                ocr_text += pytesseract.image_to_string(image)
            text = ocr_text
        except Exception as ocr_e:
            logger.error(f"OCR extraction failed: {str(ocr_e)}")

    if not text.strip():
        logger.warning("No text could be extracted from the PDF")
    return text if text is not None else ""

def extract_text_from_docx(docx_path):
    """Extract text from DOCX."""
    logger.debug(f"Extracting text from DOCX: {docx_path}")
    try:
        doc = docx.Document(docx_path)
        return " ".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        logger.error(f"Error reading DOCX: {str(e)}")
        return ""

def extract_text_from_txt(txt_path):
    """Extract text from TXT file."""
    logger.debug(f"Extracting text from TXT: {txt_path}")
    try:
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                with open(txt_path, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue
        logger.warning("Could not decode text file with common encodings")
        return ""
    except Exception as e:
        logger.error(f"Error reading TXT file: {str(e)}")
        return ""

def extract_text_from_file(file_path):
    """Determine file type and extract text accordingly."""
    logger.info(f"Extracting text from file: {file_path}")
    if file_path.endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        text = extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        text = extract_text_from_txt(file_path)
    else:
        logger.warning(f"Unsupported file type: {file_path}")
        text = ""
    return text if text is not None else ""

def preprocess_text(text):
    """Preprocess text: lowercase, remove punctuation/numbers, tokenize, remove stopwords, and lemmatize."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    try:
        # Limit text length to prevent spaCy memory issues
        MAX_SPACY_LENGTH = 100000
        if len(text) > MAX_SPACY_LENGTH:
            logger.warning(f"Text too long for spaCy processing, truncating to {MAX_SPACY_LENGTH} chars")
            text = text[:MAX_SPACY_LENGTH]
            
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct]
        return " ".join(tokens)
    except Exception as e:
        logger.error(f"Error in text preprocessing: {str(e)}")
        # Fallback to simple preprocessing
        tokens = [t.lower() for t in word_tokenize(text) if t.lower() not in stop_words]
        return " ".join(tokens)

def extract_dynamic_keywords(text, max_keywords=15):
    """
    Dynamically extract important keywords from the job description using TF-IDF.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return []
    
    processed_text = preprocess_text(text)
    if not processed_text:
        return []
        
    # Extract single words and bigrams
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=50)
        tfidf_matrix = vectorizer.fit_transform([processed_text])
        scores = tfidf_matrix.toarray()[0]
        feature_names = vectorizer.get_feature_names_out()
        keyword_scores = {feature_names[i]: scores[i] for i in range(len(feature_names))}
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        keywords = [kw for kw, score in sorted_keywords[:max_keywords]]
        
        # Add important industry terms and skills that might not be captured by TF-IDF
        if not keywords:
            return []
            
        # Extract keywords from named entities
        try:
            doc = nlp(text[:100000])  # Limit text length for spaCy
            entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "EVENT"]]
            keywords.extend([e for e in entities if e not in keywords][:5])
        except Exception as e:
            logger.warning(f"Named entity extraction failed: {str(e)}")
            
        return keywords[:max_keywords]
    except Exception as e:
        logger.error(f"Error extracting dynamic keywords: {str(e)}")
        # Fallback to simple word frequency
        words = word_tokenize(processed_text)
        word_freq = {}
        for word in words:
            if word not in word_freq:
                word_freq[word] = 1
            else:
                word_freq[word] += 1
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10]]

def identify_resume_sections(resume_text):
    """Identify common resume sections and their presence."""
    if not resume_text or not isinstance(resume_text, str):
        return {}
        
    resume_lower = resume_text.lower()
    sections = {
        'contact_info': False,
        'education': False,
        'experience': False,
        'skills': False,
        'projects': False,
        'certifications': False,
        'summary': False,
        'achievements': False,
    }
    
    # Check for contact info
    email_pattern = r'[\w\.-]+@[\w\.-]+'
    phone_pattern = r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    linkedin_pattern = r'linkedin\.com/in/[\w-]+'
    
    sections['contact_info'] = (
        re.search(email_pattern, resume_lower) is not None or
        re.search(phone_pattern, resume_lower) is not None or
        re.search(linkedin_pattern, resume_lower) is not None
    )
    
    # Check for common section headings
    section_patterns = {
        'education': r'\b(education|academic|degree|university|college|school)\b',
        'experience': r'\b(experience|employment|work history|professional|career)\b',
        'skills': r'\b(skills|expertise|competencies|proficiencies|abilities)\b',
        'projects': r'\b(projects|portfolio|works|implementations)\b',
        'certifications': r'\b(certifications|certificates|credentials|qualifications)\b',
        'summary': r'\b(summary|profile|objective|about me|overview)\b',
        'achievements': r'\b(achievements|accomplishments|awards|honors|recognition)\b',
    }
    
    for section, pattern in section_patterns.items():
        sections[section] = re.search(pattern, resume_lower) is not None
        
    return sections

def check_ats_formatting(resume_text):
    """Check resume formatting for ATS compatibility."""
    issues = []
    tips = []
    
    if not resume_text or not isinstance(resume_text, str):
        return {"issues": ["No resume text provided"], "tips": []}
    
    # Check for contact info
    if not re.search(r'[\w\.-]+@[\w\.-]+', resume_text):
        issues.append("Missing or improperly formatted email address")
        tips.append("Include a professional email address at the top of your resume")
    
    phone_pattern = r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    if not re.search(phone_pattern, resume_text):
        issues.append("Missing or improperly formatted phone number")
        tips.append("Include a phone number with area code")
        
    # Check for section headers
    sections = identify_resume_sections(resume_text)
    missing_sections = [section.replace('_', ' ').title() for section, present in sections.items() if not present]
    
    if len(missing_sections) > 0:
        if len(missing_sections) > 2:
            issues.append(f"Missing important sections: {', '.join(missing_sections[:2])} and others")
        else:
            issues.append(f"Missing important sections: {', '.join(missing_sections)}")
        tips.append("Include clearly labeled sections for Education, Experience, and Skills")
    
    # Check for bullet points
    if not re.search(r'•|\*|[\-–—]|\d+\.', resume_text):
        issues.append("No bullet points detected")
        tips.append("Use bullet points to highlight achievements and responsibilities")
    
    # Check for length (if too long)
    words = len(re.findall(r'\b\w+\b', resume_text))
    if words > 1000:
        issues.append(f"Resume may be too long ({words} words)")
        tips.append("Keep resume concise, ideally 1-2 pages (500-700 words)")
    
    # Check for problematic characters
    if re.search(r'[^\x00-\x7F]', resume_text):
        issues.append("Contains non-standard characters that may cause ATS issues")
        tips.append("Avoid special characters, emojis, and non-standard symbols")
    
    # Check for tables/columns (rough heuristic)
    lines = resume_text.split('\n')
    tab_count = sum(1 for line in lines if '\t' in line)
    if tab_count > 5:
        issues.append("Possible table or column formatting detected")
        tips.append("Avoid tables, columns, and text boxes that ATS may not parse correctly")
    
    return {
        "issues": issues,
        "tips": tips
    }

def calculate_ats_score(resume_text, job_description):
    logger.info("Calculating ATS score")
    # Ensure inputs are strings and handle None values
    resume_text = resume_text if isinstance(resume_text, str) else ""
    job_description = job_description if isinstance(job_description, str) else ""

    logger.debug(f"Resume text length: {len(resume_text)}")
    logger.debug(f"Job description length: {len(job_description)}")

    if not resume_text.strip():
        return {
            "error": "No text could be extracted from the resume",
            "overall_score": 0,
            "feedback": [{
                "type": "danger",
                "title": "Empty resume",
                "description": "No text could be extracted from your resume. Please check the file format and try again."
            }]
        }
    
    if not job_description.strip():
        logger.warning("No job description provided. Using generic placeholder.")
        job_description = "generic resume evaluation"

    # Safely truncate text before processing
    MAX_TEXT_LENGTH = 100000
    resume_text_truncated = resume_text[:MAX_TEXT_LENGTH]
    job_description_truncated = job_description[:MAX_TEXT_LENGTH]

    processed_resume = preprocess_text(resume_text_truncated)
    processed_job = preprocess_text(job_description_truncated)

    if not processed_resume:
        logger.warning("Resume preprocessing resulted in empty text")
        processed_resume = resume_text_truncated.lower()
    if not processed_job:
        logger.warning("Job description preprocessing resulted in empty text")
        processed_job = job_description_truncated.lower()

    # Dynamic keyword extraction
    dynamic_keywords = extract_dynamic_keywords(job_description_truncated)
    logger.debug(f"Dynamic keywords: {dynamic_keywords}")

    # --- Refined Keyword Matching with Fuzzy Scoring ---
    # Tokenize the resume text for improved matching.
    resume_tokens = resume_text_truncated.lower().split()
    keyword_hits = 0.0
    keyword_match_details = {}

    # Set a fuzzy matching threshold (e.g., 80% similarity)
    FUZZY_THRESHOLD = 0.8

    for keyword in dynamic_keywords:
        # Check for exact match using regex with word boundaries
        exact_pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(exact_pattern, resume_text_truncated.lower()):
            keyword_hits += 1.0
            keyword_match_details[keyword] = "exact match"
        else:
            # Use difflib to get close matches from resume tokens
            close_matches = difflib.get_close_matches(keyword, resume_tokens, n=1, cutoff=FUZZY_THRESHOLD)
            if close_matches:
                # Compute the similarity ratio as a fraction between 0 and 1.
                ratio = difflib.SequenceMatcher(None, keyword, close_matches[0]).ratio()
                # Add a fractional score (e.g., ratio scaled to maximum of 1.0)
                keyword_hits += ratio
                keyword_match_details[keyword] = f"fuzzy match ({ratio:.2f})"
            else:
                # No match found
                keyword_match_details[keyword] = "no match"
    # Calculate dynamic keyword score as a percentage
    dynamic_keyword_score = (keyword_hits / len(dynamic_keywords)) * 100 if dynamic_keywords else 50
    logger.debug(f"Keyword hits: {keyword_hits}, Dynamic Keyword Score: {dynamic_keyword_score:.1f}")

    # TF-IDF cosine similarity remains unchanged
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
        if processed_resume and processed_job:
            tfidf_matrix = vectorizer.fit_transform([processed_resume, processed_job])
            tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
        else:
            tfidf_similarity = 50
    except Exception as e:
        logger.error(f"Error calculating TF-IDF similarity: {str(e)}")
        tfidf_similarity = 50

    # Semantic similarity using spaCy remains unchanged
    try:
        MAX_SPACY_SIZE = 50000
        resume_sample = resume_text_truncated[:MAX_SPACY_SIZE].strip()
        job_sample = job_description_truncated[:MAX_SPACY_SIZE].strip()
        
        if resume_sample and job_sample:
            resume_doc = nlp(resume_sample)
            job_doc = nlp(job_sample)
            semantic_similarity = resume_doc.similarity(job_doc) * 100
        else:
            semantic_similarity = 50
    except Exception as e:
        logger.error(f"Error calculating semantic similarity: {str(e)}")
        semantic_similarity = 50

    # Formatting Score: if no issues, 100; else deduct 10 points per issue.
    formatting_check = check_ats_formatting(resume_text)
    formatting_issues = formatting_check["issues"]
    formatting_tips = formatting_check["tips"]
    formatting_score = 100 if not formatting_issues else max(0, 100 - (len(formatting_issues) * 10))
    
    # Section Coverage Score: perfect if all expected sections are present.
    sections = identify_resume_sections(resume_text)
    section_score = (sum(1 for present in sections.values() if present) / len(sections)) * 100

    # Detailed weighted overall score calculation
    overall_score = (
        dynamic_keyword_score * 0.40 +
        tfidf_similarity * 0.25 +
        semantic_similarity * 0.15 +
        formatting_score * 0.15 +
        section_score * 0.05
    )
    overall_score = max(0, min(overall_score, 100))

    logger.info(f"Component Scores: Dynamic Keyword={dynamic_keyword_score:.1f}, TF-IDF={tfidf_similarity:.1f}, Semantic={semantic_similarity:.1f}, Formatting={formatting_score:.1f}, Section={section_score:.1f}")
    logger.info(f"Overall ATS Score: {overall_score:.1f}")

    # Generate feedback as before (unchanged)
    feedback = []
    if dynamic_keywords:
        missing_keywords = [kw for kw in dynamic_keywords if keyword_match_details.get(kw, "") in ["no match"]]
        partial_matches = [kw for kw, detail in keyword_match_details.items() if "fuzzy match" in detail]
        exact_matches = [kw for kw, detail in keyword_match_details.items() if detail == "exact match"]
        
        if missing_keywords:
            feedback.append({
                "type": "danger",
                "title": "Missing key phrases",
                "description": f"Missing important terms: {', '.join(missing_keywords[:5])}" + 
                               (f" and {len(missing_keywords) - 5} more" if len(missing_keywords) > 5 else "")
            })
        if partial_matches:
            feedback.append({
                "type": "warning",
                "title": "Partial keyword matches",
                "description": f"These terms appear but may need more emphasis: {', '.join(partial_matches[:3])}" +
                               (f" and {len(partial_matches) - 3} more" if len(partial_matches) > 3 else "")
            })
        if exact_matches:
            feedback.append({
                "type": "success",
                "title": "Strong keyword matches",
                "description": f"Your resume includes {len(exact_matches)} key terms from the job description"
            })
    else:
        feedback.append({
            "type": "info",
            "title": "Generic evaluation",
            "description": "No job description provided or keywords extraction failed. Using general resume evaluation."
        })
    
    similarity_avg = (tfidf_similarity + semantic_similarity) / 2
    if similarity_avg > 75:
        feedback.append({
            "type": "success",
            "title": "Strong content alignment",
            "description": "Your resume content is well-aligned with the job requirements."
        })
    elif similarity_avg > 50:
        feedback.append({
            "type": "warning",
            "title": "Moderate content alignment",
            "description": "Your resume partially aligns with the job description. Consider tailoring it more specifically."
        })
    else:
        feedback.append({
            "type": "danger",
            "title": "Low content alignment",
            "description": "Your resume doesn't align well with this job description. Consider a significant revision."
        })

    if formatting_issues:
        feedback.append({
            "type": "warning",
            "title": "ATS formatting issues",
            "description": formatting_issues[0] + (f" and {len(formatting_issues)-1} more issues" if len(formatting_issues) > 1 else "")
        })
        if formatting_tips:
            feedback.append({
                "type": "info",
                "title": "ATS optimization tips",
                "description": "; ".join(formatting_tips[:3])
            })
    else:
        feedback.append({
            "type": "success",
            "title": "ATS-friendly formatting",
            "description": "Your resume is well-structured for ATS parsing."
        })

    result = {
        "overall_score": round(overall_score, 1),
        "dynamic_keyword_score": round(dynamic_keyword_score, 1),
        "tfidf_similarity": round(tfidf_similarity, 1),
        "semantic_similarity": round(semantic_similarity, 1),
        "formatting_score": round(formatting_score, 1),
        "section_score": round(section_score, 1),
        "feedback": feedback,
        "extracted_job_keywords": dynamic_keywords,
        "matched_keyword_count": keyword_hits,
        "total_job_keywords": len(dynamic_keywords),
        "formatting_issues": formatting_issues[:5],
        "formatting_tips": formatting_tips[:5],
        "keyword_matches": keyword_match_details
    }
    
    logger.info(f"ATS score calculation complete: {result['overall_score']}")
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    try:
        logger.info("Received analyze request")
        if 'file' not in request.files:
            logger.warning("No file part in request")
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        job_description = request.form.get('job_description', '') or ""  # Ensure it's never None

        if file.filename == '':
            logger.warning("No file selected")
            return jsonify({'error': 'No selected file'}), 400

        if not allowed_file(file.filename):
            logger.warning(f"File type not allowed: {file.filename}")
            return jsonify({'error': 'File type not allowed. Please upload PDF, DOCX, or TXT files.'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        if not os.path.exists(file_path):
            logger.error(f"Failed to save file to {file_path}")
            return jsonify({'error': 'Failed to save file'}), 500

        resume_text = extract_text_from_file(file_path)
        
        # Ensure resume_text is a string
        if resume_text is None:
            resume_text = ""
            
        if not resume_text.strip():
            logger.warning("No text extracted from file")
            result = {
                'error': 'Could not extract text from file',
                'overall_score': 0,
                'feedback': [{
                    'type': 'danger',
                    'title': 'Text extraction failed',
                    'description': 'No text could be extracted from your resume. Please check the file format and try again.'
                }]
            }
        else:
            # Calculate ATS score with proper error handling
            try:
                result = calculate_ats_score(resume_text, job_description)
            except Exception as score_e:
                logger.error(f"Error calculating ATS score: {str(score_e)}")
                result = {
                    'error': 'Error analyzing resume',
                    'overall_score': 0,
                    'feedback': [{
                        'type': 'danger',
                        'title': 'Analysis error',
                        'description': f'An error occurred during analysis: {str(score_e)}'
                    }]
                }

        # Clean up the temporary file
        try:
            os.remove(file_path)
            logger.debug(f"Deleted temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not delete temporary file: {str(e)}")

        # Validate response before returning
        try:
            # Ensure we don't have any undefined values that might cause JSON issues
            json_response = json.dumps(result)
            return jsonify(result)
        except Exception as json_error:
            logger.error(f"Error serializing response to JSON: {str(json_error)}")
            # Return a safe response
            return jsonify({
                'error': 'Error processing results',
                'overall_score': 0,
                'feedback': [{
                    'type': 'danger',
                    'title': 'Processing error',
                    'description': 'An error occurred while processing the results.'
                }]
            })

    except Exception as e:
        logger.error(f"Uncaught exception: {str(e)}")
        traceback.print_exc()
        # Clean up if file was created
        if 'file_path' in locals() and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
        return jsonify({
            'error': f'Server error: {str(e)}',
            'overall_score': 0,
            'feedback': [{
                'type': 'danger',
                'title': 'Server error',
                'description': 'An unexpected error occurred. Please try again later.'
            }]
        }), 500

if __name__ == '__main__':
    logger.info("Starting ATS Resume Analyzer application")
    app.run(debug=True)
