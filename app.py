import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
import os
import torch
from transformers import AutoModel
from sentence_transformers import SentenceTransformer

# LangChain imports (updated for v0.2+)
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub

from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


def load_embedding_model():
    try:
        model = SentenceTransformer("hkunlp/instructor-xl")
        return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

def get_text_from_file(files):
    text = ""
    for doc in files:
        if doc.name.endswith(".pdf"):
            pdf_reader = PdfReader(doc)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        elif doc.name.endswith(".docx"):
            docx_reader = Document(doc)
            for para in docx_reader.paragraphs:
                text += para.text + "\n"
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator="\n", length_function=len)
    chunks = text_splitter.split_text(raw_text)
    return [chunk for chunk in chunks if chunk.strip()]

def get_vectorstore(text_chunks):
    embeddings = load_embedding_model()
    if embeddings is None:
        return None
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-xxl",
        model_kwargs={"temperature": 0.1, "max_length": 512},
        pipeline_kwargs={"task": "text2text-generation", "max_length": 512}
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
        memory=memory,
        verbose=True
    )
    return conversation_chain

def handle_user_input(user_input):
    if st.session_state.conversation:
        response = st.session_state.conversation({"question": user_input})
        st.session_state.chat_history = response['chat_history']
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        st.session_state.conversation.memory.add_user_message(user_input)
        st.session_state.conversation.memory.add_ai_message(response['answer'])
        st.text_area("Response:", value=response['answer'], height=300, key="response")
    else:
        st.error("Please upload documents and process them first before asking questions.")

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Documents", page_icon="📚", layout="wide")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "response" not in st.session_state:
        st.session_state.response = ""

    st.header("Chat with Your Documents 📚")

    user_questions = st.text_input("Enter your question:")
    if user_questions:
        handle_user_input(user_questions)

    st.text_area("Response:", height=300)
    st.button("Submit")
    st.button("Clear")

    st.write(user_template.replace("{{MSG}}", "hello Machine"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "hello GURLLL"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your Documents")
        uploaded_files = st.file_uploader("Upload your documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)

        if uploaded_files:
            for file in uploaded_files:
                st.write(file.name)

        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_text_from_file(uploaded_files)
                text_chunks = get_text_chunks(raw_text)
                if not text_chunks:
                    st.error("No valid text found in the uploaded files.")
                    return
                vectorstore = get_vectorstore(text_chunks)
                if vectorstore:
                    st.success("Vectorstore created successfully!")
                else:
                    st.error("Failed to create vectorstore. Check model loading.")
                st.session_state.conversation = get_conversation_chain(vectorstore)

        st.sidebar.button("Add documents")
        st.sidebar.button("Remove documents")
        st.sidebar.button("Save chat history")

if __name__ == "__main__":
    main()
