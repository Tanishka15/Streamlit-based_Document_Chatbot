# Chat with Your Documents

This project is a Streamlit-powered chatbot application that allows you to upload and chat with your own documents (PDF, DOCX, TXT). Using LangChain, FAISS, and transformer-based models, it enables contextual question-answering based on the uploaded documents.

---

## Features

- Upload multiple `.pdf`, `.docx`, and `.txt` files
- Automatic text splitting and vector embedding
- Conversational memory using LangChain
- Powered by `Flan-T5-XXL` from HuggingFace for natural responses
- Document search using FAISS vector store
- Dynamic chatbot interface with message history

---

## How It Works

1. Upload one or more documents via the sidebar.
2. The documents are parsed and split into manageable chunks.
3. Each chunk is converted into a vector embedding using `Instructor-XL`.
4. These embeddings are stored in a FAISS vector database.
5. A ConversationalRetrievalChain is created using a language model (`flan-t5-xxl`) that:
    - Remembers past questions
    - Retrieves relevant chunks for each user question
6. The system returns accurate, context-aware answers from the document contents.

---

## Project Structure

```
chat-with-documents/
├── app.py                   # Main Streamlit application
├── htmlTemplates.py         # HTML templates for chatbot UI
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## Setup Instructions

### Prerequisites

- Python 3.9+
- Git
- A HuggingFace account with API token

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chat-with-documents.git
cd chat-with-documents

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your HuggingFace token
touch .env
echo "HUGGINGFACEHUB_API_TOKEN=your_token_here" > .env
```

### Run the App

```bash
streamlit run app.py
```

---

## Example Use Case

1. Upload a research paper or company policy document.
2. Ask: "What are the main conclusions?" or "Who is responsible for compliance?"
3. Get intelligent responses grounded in the actual content.

---

## Tech Stack

- Streamlit
- LangChain
- FAISS
- HuggingFace Transformers
- Sentence-Transformers
- PyPDF2
- python-docx

---

## Future Improvements

- Add support for images/OCR
- Upload and manage sessions per user
- Download or export chat history
- Fine-tune model for domain-specific data

---

## License

MIT License © 2025 Tanishka Rajratna

---

## Author

Tanishka Rajratna  

```
