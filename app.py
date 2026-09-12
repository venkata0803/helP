import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv

# Set page configuration and load API key
st.set_page_config(page_title="helP - PDF Helper", layout="wide", page_icon="📄")
load_dotenv()
os.environ['NVIDIA_API_KEY'] = "API_KEY"

# NVIDIA green theme styling
st.markdown(
    """
    <style>
    body {background-color: #1C2D2A; color: #ecf0f1;}
    .stTextInput input, .stButton>button, .stTextArea textarea, .stExpander {
        background-color: #2B4C48; color: #ecf0f1; border-radius: 5px; border: none;
    }
    .stButton>button:hover {background-color: #59D361;}
    h1 {color: #76B900; font-size: 3em; text-align: center;}
    .question-answer-box {padding: 10px; background-color: #344F4C; border-radius: 8px; margin-bottom: 10px;}
    </style>
    """, unsafe_allow_html=True
)

# Initialize vector embedding function
def vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner("🔄 Initializing document embeddings..."):
            st.session_state.embeddings = NVIDIAEmbeddings()
            st.session_state.loader = PyPDFDirectoryLoader("./books")
            st.session_state.docs = st.session_state.loader.load()

            # Adjust chunk size for better context
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])

            # Create vector store for embeddings
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("Document embeddings initialized successfully!")

# Sidebar - Settings and Actions
with st.sidebar:
    st.header("Settings & Actions")

    # Start New Chat button
    if st.button("Start New Chat"):
        st.session_state.chat_history = []
        st.success("New chat started!")

    # Reload Documents button
    if st.button("Reload Documents"):
        vector_embedding()
        st.info("Documents reloaded successfully!")

    # Clear Session button
    if st.button("Clear Session"):
        st.session_state.clear()
        st.warning("Session reset. Reinitialize documents to continue.")

# Main page layout
st.markdown("<h1>helP</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #95a5a6;'>Your PDF Helper Chatbot</h5>", unsafe_allow_html=True)

# Initialize the language model
llm = ChatNVIDIA(model="meta/llama-3.2-3b-instruct")

# Refined prompt template for querying
prompt = ChatPromptTemplate.from_template("""
Use the context from the document below to answer the following question.
If the document does not contain the answer, respond with "The information is not available in the document."
<context>
{context}
</context>
Question: {input}
""")

# Input field for questions
prompt1 = st.text_input("Ask a question based on the uploaded documents", key="user_question")

# Initialize document embeddings if button clicked
if st.button("Initialize Document Reading"):
    vector_embedding()
    st.success("You can start asking questions now!")

# Maintain chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Process the question if provided
if prompt1 and "vectors" in st.session_state:
    with st.spinner("Getting your answer..."):
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start_time = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        response_time = time.process_time() - start_time

        # Store user input and response in chat history
        st.session_state.chat_history.append(("User", prompt1))
        answer = response.get('answer', "The information is not available in the document.")
        st.session_state.chat_history.append(("Assistant", answer))

        # Display the question and answer in chat format immediately
        for speaker, message in st.session_state.chat_history:
            st.write(f"<div class='question-answer-box'><strong>{speaker}:</strong> {message}</div>", unsafe_allow_html=True)

        # Relevant document context in an expander
        with st.expander("🔍 Relevant Document Context"):
            for doc in response.get("context", []):
                st.write(doc.page_content)
                st.write("--------------------------------")
