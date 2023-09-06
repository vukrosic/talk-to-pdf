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
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEYY")
#openai.api_key = OPENAI_API_KEY
#top_k = 2
# Function to add instructions text

# Initialize the global variables
OPENAI_API_KEY = ''
PINECONE_API_KEY = ''
PINECONE_API_ENV = ''
PINECONE_INDEX_NAME = 'test'
top_k = 3
chunk_size = 1000

# Function to add instructions text
def instructions():
    global top_k  # Declare top_k as a global variable
    global chunk_size  # Declare chunk_size as a global variable
    st.markdown(
        """
        **Instructions:**

        1. **OpenAI API Key:** You can obtain your OpenAI API key from the OpenAI platform. [Get API Key](https://platform.openai.com/account/api-keys)
        2. **Pinecone API Key:** You can obtain your Pinecone API key from the Pinecone console. [Get API Key](https://www.pinecone.io/)
        3. **Pinecone Environment:** Enter the name of your Pinecone environment. [Get Environment](https://www.pinecone.io/)
        4. **Pinecone Index:** Create a new pinecone index with dimentions = 1536 and metric = cosine. Then set the index name. [Create Index](https://www.pinecone.io/)

        _Although these parameters will reset when you reload the page, the data in the pinecone database will not, so you don't need to "Embed To Database" the same data again, unless you are changing it. You will only need to fill OpenAI API key if you're not embedding new data._

        Pinecone is free. You can also use free OpenAI API but it's limited to 3 requests per minute. 
        I recommend upgrading it. 
        
        _Embedding cost: around \$1 per 3000 pages (\$0.0006 per 1000 words)_
        
        _GPT generation cost: around \$0.002 per 1,000 words_
         
        It will cost you less than cent per question. You can track your costs at https://platform.openai.com/account/usage.  
        
        _This is all you need to know, continue reading for additional explanations._


        The PDFs and text are separated into 1000 character chunks (142 - 250 words) and stored into a database. When you ask a question, it will find the top 3 most relevant chunks
        and send them to GPT-3.5 (ChatGPT) together with your question to generate the answer.


        This is an example prompt that is sent to GPT:

        ----------------------------------
        
        Relevant Documents:

        {relevant_documents_text}
        
        User Question: {question}
        
        Answer:
        
        ----------------------------------

        Variable **'{question}'** contains the user's question, for example: 
        
        _What is the role of the stem inside the pseudostem in banana plant development?_

        Variable **'{relevant_documents_text}'** contains most relevant chunks of text, ordered by relevancy, for example:

        _stops producing new leaves and begins to form a flower spike or inflorescence. A stem develops which grows up inside the pseudostem, carrying the immature inflorescence until eventually it emerges at the top.[19] Each pseudostem normally produces a single inflorescence, also known as the "banana heart". (More are sometimes produced; an exceptional plant in the Philippines produced five.[20]) After f_
        
        _iting, the pseudostem dies, but offshoots will normally have developed from the base, so that the plant as a whole is perennial. In the plantation system of cultivation, only one of the offshoots will be allowed to develop in order to maintain spacing.[21] The inflorescence contains many bracts (sometimes incorrectly referred to as petals) between rows of flowers. The female flowers (which can develo_

        _ing cluster, made up of tiers (called "hands"), with up to 20 fruit to a tier. The hanging cluster is known as a bunch, comprising 3–20 tiers, or commercially as a "banana stem", and can weigh 30–50 kilograms (66–110 lb). Individual banana fruits (commonly known as a banana or "finger") average 125 grams (4+1⁄2 oz), of which approximately 75% is water and 25% dry matter (nutrient table, lower right)._

        
        ----------------------------------

        GPT will generate an answer based on the prompt we sent.

        **'chunk_size' parameter** is the number of characters in each chunk. Default is 1000, the more characters the more information will be sent to GPT, 
        but will also cost more, although cost is still very low.

        **'top_k' parameter** is the number of most relevant chunks to retrieve and send to GPT. Default is 3, the more chunks the more information will be sent to GPT,
        but will also cost more, although cost is still very low.

        """
    )
    top_k = st.number_input("Set 'top_k' parameter", min_value=1, step=1, value=top_k)  # Assign the value to the global variable
    chunk_size = st.number_input("Set 'chunk_size' parameter", min_value=100, max_value=10000, step=250, value=chunk_size)


# Function to add a button to toggle instructions
def toggle_instructions_button():
    show_instructions = st.checkbox("Show Instructions")
    if show_instructions:
        instructions()  # Display instructions when the checkbox is checked




# Initialize Pinecone
def initialize_pinecone():
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    return pinecone


def upload_data_to_pinecone(texts):
    # Initialize Pinecone
    pinecone = initialize_pinecone()
    # Initialize Langchain embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # Convert and upload data as tuples (ID, vector)
    data_to_upload = [(str(i), embeddings.embed_query(text), {"text": text}) for i, text in enumerate(texts)]
    # Upload the data to Pinecone
    index = pinecone.Index(PINECONE_INDEX_NAME)
    index.delete(delete_all=True)
    index.upsert(data_to_upload)



# Answer questions based on data in Pinecone
def answer_question(question):
    # Initialize Langchain embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Convert the user's question into an embedding
    question_embedding = embeddings.embed_query(question)

    # Search for the most similar embeddings in Pinecone
    index = pinecone.Index(PINECONE_INDEX_NAME)

    try:
        results = index.query(queries=[question_embedding], top_k=top_k, include_metadata=True)
    except Exception as e:
        return "Error, please try again. If this persists, contact me at vukrosic1@gmail.com"
    # print(str(results['results'][0]['matches']['metadata']['text']))
    # Access the matches correctly
    matches = results['results'][0]['matches']
    relevant_documents = [match['metadata']['text'] for match in matches]

    # Concatenate relevant documents into a single text
    relevant_documents_text = "\n\n".join(relevant_documents)

    # Create a chat prompt with relevant documents and the user's question
    chat_prompt = f"Relevant Documents:\n{relevant_documents_text}\n\nUser Question: {question}\nAnswer:"
    openai.api_key = OPENAI_API_KEY
    # Generate an answer using GPT-3.5 Turbo
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": chat_prompt}
        ]
    )
    answer = response.choices[0].message.content
    return answer

def main():
    # Define your Pinecone index name
    texts = []
    global OPENAI_API_KEY
    global PINECONE_API_KEY
    global PINECONE_API_ENV
    global PINECONE_INDEX_NAME
    initialize_pinecone()



    # Access the API keys and other user-defined parameters using text input fields
    OPENAI_API_KEY = st.text_input("Enter your [OpenAI API Key](https://platform.openai.com/account/api-keys)")
    PINECONE_API_KEY = st.text_input("Enter your [Pinecone API Key](https://www.pinecone.io/)")
    PINECONE_API_ENV = st.text_input("Enter your [Pinecone Environment](https://www.pinecone.io/)")
    PINECONE_INDEX_NAME = st.text_input("Enter your [Pinecone Index Name](https://www.pinecone.io/)")

    # Add the toggle button to show or hide instructions
    toggle_instructions_button()
    # Streamlit App
    st.title("Answer questions from text")

    # File Upload
    pasted_text = st.text_area("Paste text here:", "")
    uploaded_files = st.file_uploader("And / or upload PDF files:", type=["pdf"], accept_multiple_files=True)
    st.info("It's best to paste text. PDF reading is a lot slower. Use websites like https://www.pdf2go.com/pdf-to-text to convert PDF to text.")


    if st.button("Embed To Database"):
        # Initialize a variable to store the extracted text
        extracted_text = ""
        # Iterate through each uploaded PDF file
        for uploaded_file in uploaded_files:
            pdf_reader = PdfReader(uploaded_file)
            # Iterate through each page and extract text
            for page in pdf_reader.pages:
                extracted_text += page.extract_text()
        extracted_text += pasted_text
        # Split the text into smaller chunks for embedding and Pinecone upload
        texts = [extracted_text[i:i + chunk_size] for i in range(0, len(extracted_text), chunk_size)]
        # Upload the extracted text to Pinecone
        upload_data_to_pinecone(texts)

    # Streamlit App (continued)
    st.header("Ask a Question")
    relevant_documents = ''
    # User input for question
    user_question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        # Retrieve answers from Pinecone based on the user's question
        relevant_documents = answer_question(user_question)
    st.write(relevant_documents)
if __name__ == "__main__":
    main()