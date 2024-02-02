
import streamlit as st
import openai
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from openai import OpenAI

def convert_test_conversation(train_sentence, system_message=None):
    # Initializing the messages list
    messages = []

    # Including the system message if provided
    if system_message:
        messages.append({
            "role": "system",
            "content": system_message
        })
        # Formatting the message
        message = {
            "role": "user",
            "content": "Extract the useful information for the abstract: "+ train_sentence
        }
        messages.append(message)
    return messages

def generate_response(openai_api_key, query_text):
    # Load document if file is uploaded
    if query_text is not None:
        # documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        # texts = text_splitter.create_documents(documents)
        # Select embeddings
        system_message = "You are a helpful literature analyst, you can help extract the useful information from an abstract for the following topics: 1. input data used, 2. features of the input data used, 3. model used, 4. output given"
        test_record = convert_test_conversation(query_text, system_message=system_message)
        fine_tuned_model_id = 'ft:gpt-3.5-turbo-0613:weclouddata:fiqa:8nLzwa3N'
        response = client.chat.completions.create(model=fine_tuned_model_id, messages=item, temperature=0, max_tokens=500)
        # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Create a vectorstore from documents
        # db = Chroma.from_documents(texts, embeddings)
        # Create retriever interface
        # retriever = db.as_retriever()
        # Create QA chain
        # qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
        return response.choices[0].message.content

# Page title
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# File upload
# uploaded_file = st.file_uploader('Upload an article', type='txt')
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide the abstract you want to process for Damage Identification task.')

# Form input and query
result = []
with st.form('myform', clear_on_submit=False):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not query_text)
    submitted = st.form_submit_button('Submit', disabled=not query_text)
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)
