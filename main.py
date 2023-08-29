import streamlit as st
import os
from PyPDF2 import PdfReader
import pinecone
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load the environment variables from .env file
load_dotenv()

# Access the API key using the environment variable name
OPENAI_API_KEY = os.getenv("OPENAI_API_KEYY")

# Initialize Pinecone
def initialize_pinecone():
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    return pinecone


def upload_data_to_pinecone(texts, index_name):
    # Initialize Pinecone
    pinecone = initialize_pinecone()
    # Initialize Langchain embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # Convert and upload data as tuples (ID, vector)
    data_to_upload = [(str(i), embeddings.embed_query(text)) for i, text in enumerate(texts)]
    # Upload the data to Pinecone
    index = pinecone.Index(index_name)
    index.delete(delete_all=True)
    index.upsert(data_to_upload)



# Answer questions based on data in Pinecone
def answer_question(question, index_name, texts):
    # Initialize Langchain embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Convert the user's question into an embedding
    question_embedding = embeddings.embed_query(question)

    # Search for the most similar embeddings in Pinecone
    with pinecone.Index(index_name) as index:
        results = index.query(queries=[question_embedding], top_k=5)

    # Access the matches correctly
    matches = results['results'][0]['matches']

    relevant_documents = [texts[int(match['id'])] for match in matches]

    return relevant_documents

def main():
    # Define your Pinecone index name
    index_name = "test"  # Replace with your desired index name
    texts = []

    # Streamlit App
    st.title("PDF Text Extractor and Question Answering")

    # File Upload
    uploaded_files = st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=True)

    if st.button("Analyze Files"):
        # Initialize a variable to store the extracted text
        extracted_text = ""
        # Iterate through each uploaded PDF file
        for uploaded_file in uploaded_files:
            pdf_reader = PdfReader(uploaded_file)
            # Iterate through each page and extract text
            for page in pdf_reader.pages:
                extracted_text += page.extract_text()

        # Split the text into smaller chunks for embedding and Pinecone upload
        chunk_size = 1000  # You can adjust this based on your needs
        texts = [extracted_text[i:i + chunk_size] for i in range(0, len(extracted_text), chunk_size)]

        # Upload the extracted text to Pinecone
        upload_data_to_pinecone(texts, index_name)

    # Streamlit App (continued)
    st.header("Ask a Question")

    # User input for question
    user_question = st.text_input("Enter your question:")

    if st.button("Get Answer"):

        # Retrieve answers from Pinecone based on the user's question
        relevant_documents = answer_question(user_question, index_name, texts)

        # Display the relevant documents as answers
        st.subheader("Answers:")
        for document in relevant_documents:
            st.write("----------------------")
            st.write(document)

if __name__ == "__main__":
    main()