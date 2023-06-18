# Imports
import os 
import chromadb
import pypdf
import tiktoken
import streamlit as st
from streamlit_chat import message
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI

API_KEY = "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"
model_id = "gpt-3.5-turbo"

os.environ["OPENAI_API_KEY"] = API_KEY

llm=ChatOpenAI(model_name = model_id, temperature=0.2)

loaders = PyPDFLoader('docs/valuation.pdf')

index = VectorstoreIndexCreator().from_loaders([loaders])

st.title('✨ Query your Documentation ')
prompt = st.text_input("Enter your question to query your Camera Documentation ")

if prompt:
    # stuff chain type sends all the relevant text chunks from the document to LLM    
    response = index.query(llm=llm, question = prompt, chain_type = 'stuff')    

    # Write the results from the LLM to the UI
    st.write("<br><i>" + response + "</i><hr>", unsafe_allow_html=True )
    #st.write("<b>" + prompt + "</b><br><i>" + response + "</i><hr>", unsafe_allow_html=True )
