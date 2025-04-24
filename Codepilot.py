__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.docstore.document import Document

# Load environment variables
_ = load_dotenv(find_dotenv())
api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(temperature=0.2, model='llama3-70b-8192') # Increased temperature slightly for generation

# Cache FAISS index for repeated queries
@st.cache_resource
def build_faiss_index(code_input):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents([code_input])
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_documents(docs, embeddings)

# Function to generate response, handling both code-based questions and code generation
def generate_response(code, query):
    if query.lower().startswith("write a code") or query.lower().startswith("generate code"):
        # Directly ask the LLM to generate code
        return llm.invoke(query).content
    elif code.strip():
        # Process questions about the provided code
        store = build_faiss_index(code)
        retrieval_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=store.as_retriever()
        )
        return retrieval_chain.run(query)
    else:
        return "Please provide code or a code generation request."

# Streamlit UI setup
st.set_page_config(page_title="Code Copilot - Ask About Code", layout="wide")
st.title("🤖 AI Code Copilot")

with st.sidebar:
    st.markdown("## Input Code (Optional)")
    code_input = st.text_area("✍️ Paste your code here:", height=300, placeholder="Paste a code snippet you'd like help with (optional)")

query_text = st.text_area("🔍 Ask a question or request code:", placeholder="e.g., What does this function do? or Write a code to get all prime numbers from 2 to 50.", height=150)

if query_text:
    with st.spinner("🚀 Processing your request..."):
        response = generate_response(code_input, query_text)
        st.subheader("📌 Copilot Answer")
        st.code(response, language="python")
else:
    st.info("Enter a question or a code generation request to get started.")
