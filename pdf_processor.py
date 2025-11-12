import os
import sys
# Prevent TensorFlow from loading (causes AVX errors on some CPUs)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['DISABLE_TF'] = '1'
# Fix protobuf compatibility issue
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
# Block TensorFlow import to prevent AVX errors
sys.modules['tensorflow'] = None
sys.modules['tensorflow.python'] = None


from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pickle

# Lazy import streamlit to avoid import issues
def _get_streamlit():
    try:
        import streamlit as st
        return st
    except ImportError:
        return None

class PDFProcessor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_text_from_pdf(self, pdf_path: str) -> List[Document]:
        """Extract text from PDF file"""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            st = _get_streamlit()
            if st:
                st.error(f"Error processing {pdf_path}: {str(e)}")
            else:
                print(f"Error processing {pdf_path}: {str(e)}")
            return []

    def extract_text_from_uploaded_pdf(self, uploaded_file) -> List[Document]:
        """Extract text from uploaded PDF file"""
        try:
            # Save uploaded file temporarily
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Process the temporary file
            chunks = self.extract_text_from_pdf("temp.pdf")

            # Clean up
            os.remove("temp.pdf")
            return chunks
        except Exception as e:
            st = _get_streamlit()
            if st:
                st.error(f"Error processing uploaded file: {str(e)}")
            else:
                print(f"Error processing uploaded file: {str(e)}")
            return []

    def load_pdfs_from_folder(self, folder_path: str = "pdfs") -> List[Document]:
        """Load all PDFs from specified folder"""
        all_chunks = []
        st = _get_streamlit()

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            if st:
                st.warning(f"Created {folder_path} folder. Please add your PDF files there.")
            else:
                print(f"Created {folder_path} folder. Please add your PDF files there.")
            return all_chunks

        pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

        if not pdf_files:
            if st:
                st.info(f"No PDF files found in {folder_path} folder.")
            else:
                print(f"No PDF files found in {folder_path} folder.")
            return all_chunks

        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder_path, pdf_file)
            if st:
                st.info(f"Processing: {pdf_file}")
            else:
                print(f"Processing: {pdf_file}")
            chunks = self.extract_text_from_pdf(pdf_path)
            all_chunks.extend(chunks)

        return all_chunks

    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """Create FAISS vectorstore from documents"""
        if not documents:
            return None

        vectorstore = FAISS.from_documents(documents, self.embeddings)
        return vectorstore

    def save_vectorstore(self, vectorstore: FAISS, path: str = "vectorstore"):
        """Save vectorstore to disk"""
        if not os.path.exists(path):
            os.makedirs(path)
        vectorstore.save_local(path)

    def load_vectorstore(self, path: str = "vectorstore") -> Optional[FAISS]:
        """Load vectorstore from disk"""
        try:
            if os.path.exists(path):
                vectorstore = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
                return vectorstore
        except Exception as e:
            st = _get_streamlit()
            if st:
                st.error(f"Error loading vectorstore: {str(e)}")
            else:
                print(f"Error loading vectorstore: {str(e)}")
        return None
