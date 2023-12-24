from transformers import BartForConditionalGeneration, BartTokenizer
import PyPDF2

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(pdf_reader.numPages):
            text += pdf_reader.getPage(page_num).extractText()
    return text

def generate_summary_bart(document, max_length=150):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

    inputs = tokenizer.encode("Summarize: " + document, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, num_beams=5, length_penalty=2.0, no_repeat_ngram_size=3, top_k=50, top_p=0.95)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Example usage
pdf_path = 'project_feature_details.pdf'
document_text = extract_text_from_pdf(pdf_path)

# Generate summary
context_aware_summary = generate_summary_bart(document_text)
print(f"\nContext-aware Summary:\n{context_aware_summary}")
