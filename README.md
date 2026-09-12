# helP - PDF Helper Chatbot 

helP is a PDF-based question-answering chatbot built using Streamlit and Retrieval-Augmented Generation (RAG).

The application allows users to load PDF documents, create embeddings from their content, and ask questions about the documents. Relevant document sections are retrieved using FAISS and passed to an NVIDIA-powered Large Language Model to generate answers.

## Features

-  Load and process PDF documents
-  Retrieve relevant document content using semantic search
-  Answer questions using an NVIDIA-powered LLM
-  Interactive chatbot-style conversation history
-  Display relevant document context used to generate answers
-  Reload documents when required
-  Start a new chat session
-  Clear the current session
-  Display a responsive Streamlit web interface
-  NVIDIA-inspired green-themed UI

## How It Works

The application follows a Retrieval-Augmented Generation (RAG) workflow:

```text
                PDF Documents
                     │
                     ▼
            PyPDFDirectoryLoader
                     │
                     ▼
              Text Splitting
                     │
                     ▼
          NVIDIA Text Embeddings
                     │
                     ▼
              FAISS Vector Store
                     │
                     ▼
                User Question
                     │
                     ▼
                FAISS Retriever
                     │
                     ▼
          Relevant Document Context
                     │
                     ▼
             NVIDIA LLM (NIM)
                     │
                     ▼
                  Answer
