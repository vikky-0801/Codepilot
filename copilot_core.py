import os
from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

def initialize_copilot():
    """
    Initializes the copilot by loading the API key and setting up the language model.

    Returns:
        ChatGroq: The initialized language model.
    """
    load_dotenv(find_dotenv())
    api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(temperature=0.2, model='llama3-70b-8192')
    return llm

def build_faiss_index(code_input):
    """
    Builds a FAISS index for efficient retrieval of code snippets.

    Args:
        code_input (str): The input code.

    Returns:
        FAISS: The FAISS index.
    """
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents([code_input])
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_documents(docs, embeddings)

def generate_response(llm, code, query):
    """
    Generates a response based on the provided code and query.

    Args:
        llm (ChatGroq): The language model.
        code (str): The input code.
        query (str): The user query.

    Returns:
        str: The generated response.
    """
    if not code and not query:
        return "Please provide code or a code generation request."

    if query.lower().startswith("write a code") or query.lower().startswith("generate code"):
        # Directly ask the LLM to generate code
        return llm.invoke(query).content
    elif code:
        # Process questions about the provided code
        store = build_faiss_index(code)
        retrieval_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=store.as_retriever()
        )
        return retrieval_chain.run(query)
    else:
        return llm.invoke(query).content  # handles the case where only query is provided.
