import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(pdf_reader.numPages):
            text += pdf_reader.getPage(page_num).extractText()
    return text

# Function to perform keyword extraction using TF-IDF
def extract_keywords_tfidf(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))

    # Fit and transform the text
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    # Sum TF-IDF scores for each word across all sentences
    word_scores = tfidf_matrix.sum(axis=0)

    # Get indices of top-scoring words
    top_word_indices = word_scores.argsort()[0, ::-1]

    # Get the top keywords
    keywords = [feature_names[index] for index in top_word_indices[:10]]  # Adjust the number of keywords as needed

    return keywords

# Function to generate summary using sentence similarity
def generate_summary(text, user_keyword):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))

    # Fit and transform the text
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Get user's keyword TF-IDF vector
    user_keyword_vector = vectorizer.transform([user_keyword])

    # Compute cosine similarity between user's keyword and each sentence
    sim_scores = cosine_similarity(user_keyword_vector, tfidf_matrix).flatten()

    # Rank sentences based on similarity scores
    ranked_sentences = [sentences[i] for i in sim_scores.argsort()[::-1]]

    # Select top sentences for the summary
    summary = ' '.join(ranked_sentences[:5])  # Adjust the number of sentences in the summary as needed

    return summary

# Example usage
pdf_path = 'project_feature_details.pdf'
user_keyword = input("Enter the keyword you're interested in: ")

# Extract text from PDF
document_text = extract_text_from_pdf(pdf_path)

# Extract keywords using TF-IDF
keywords = extract_keywords_tfidf(document_text)

# Check if user's keyword is in extracted keywords
if user_keyword.lower() in map(str.lower, keywords):
    # Generate summary using sentence similarity
    keyword_summary = generate_summary(document_text, user_keyword)
    print(f"\nSummary for '{user_keyword}':\n{keyword_summary}")
else:
    print(f"\n'{user_keyword}' not found in extracted keywords.")
