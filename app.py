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

import streamlit as st
from pdf_processor import PDFProcessor
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Auto-load pre-built vectorstore on startup
if st.session_state.vectorstore is None:
    processor = PDFProcessor()
    vectorstore = processor.load_vectorstore()
    if vectorstore:
        st.session_state.vectorstore = vectorstore

# Page config
st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom CSS with UI improvements
st.markdown("""
<style>
    /*
    MODIFICATION: The following CSS rules have been adjusted to improve the UI based on your request.
    1. The chat input box is now taller for easier multi-line input.
    2. The large gap between the 'Ask Questions' title and the chat input (when no messages are present) has been removed.
    */

    /* Make the text area within the chat input box taller */
    .stChatInput > div > textarea {
        min-height: 6rem; /* You can adjust this value for desired height */
    }

    /* Reduce gap between title and the chat area */
    /* This rule was already present and is effective. */
    .element-container:has(h3) {
        margin-bottom: 0.5rem !important;
    }

    /*
    The previous rule for .stChatMessageContainer which set a 'min-height: 400px' has been removed.
    This was causing the large empty space. By removing it, the container will only take up space
    when chat messages are actually present.
    */
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_llm():
    """Load and cache the LLM model"""
    with st.spinner("Loading AI model... This may take a moment."):
        # Use a small, efficient model
        model_name = "microsoft/DialoGPT-small"

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
            model = AutoModelForCausalLM.from_pretrained(model_name)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                device=0 if torch.cuda.is_available() else -1
            )

            return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

@st.cache_resource
def initialize_processor():
    """Initialize PDF processor"""
    return PDFProcessor()

def main():
    # Initialize components
    processor = initialize_processor()

    # Left Sidebar for PDF management
    with st.sidebar:
        st.header("📄 Document Management")

        # Show knowledge base status
        if st.session_state.vectorstore:
            st.success("✅ Knowledge base loaded and ready!")
            st.info("📚 13.1MB of documents pre-loaded")
            st.caption("You can start asking questions immediately")
        else:
            st.error("❌ Knowledge base not found")
            st.warning("Please contact the administrator")

        st.markdown("---")

        # Optional: Show loaded documents info
        with st.expander("ℹ️ About the Documents"):
            st.write("This assistant has been pre-trained with:")
            st.write("- MasterCard Policy Documents")
            st.write("- Total size: 13.1MB")

        st.markdown("---")

        # Upload additional PDFs
        st.subheader("Upload Additional PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload additional PDF files to query"
        )

        if uploaded_files:
            if st.button("📤 Process Uploaded PDFs"):
                with st.spinner("Processing uploaded PDFs..."):
                    new_documents = []
                    for uploaded_file in uploaded_files:
                        chunks = processor.extract_text_from_uploaded_pdf(uploaded_file)
                        new_documents.extend(chunks)

                    if new_documents:
                        # Load existing vectorstore or create new one
                        existing_vectorstore = processor.load_vectorstore()

                        if existing_vectorstore and 'vectorstore' in st.session_state:
                            # Add new documents to existing vectorstore
                            new_vectorstore = processor.create_vectorstore(new_documents)
                            existing_vectorstore.merge_from(new_vectorstore)
                            st.session_state.vectorstore = existing_vectorstore
                        else:
                            # Create new vectorstore
                            vectorstore = processor.create_vectorstore(new_documents)
                            st.session_state.vectorstore = vectorstore

                        processor.save_vectorstore(st.session_state.vectorstore)
                        st.success(f"Added {len(new_documents)} new document chunks!")

    # Main content area - Title at the very top
    st.title("📚 PDF Q&A Assistant")
    st.markdown("Ask questions about your PDF documents using AI!")

    # How to Use - right below the subtitle
    st.markdown("""
    **📋 How to Use:**
    1. 📚 MasterCard PDFs are pre-loaded and ready
    2. 💬 Type your question in the chat box below
    3. 📖 Expand 'Sources' to see document references
    4. 🔄 Ask follow-up questions anytime
    """)

    st.markdown("---")

    # Chat interface - removed the subheader to reduce gap
    st.markdown("### 💬 Ask Questions")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input - right after messages with minimal gap
    prompt = st.chat_input("Ask a question about your documents...", key="chat_input")

    if prompt:
        # Check if vectorstore exists
        if st.session_state.vectorstore is None:
            # Try to load existing vectorstore
            vectorstore = processor.load_vectorstore()
            if vectorstore:
                st.session_state.vectorstore = vectorstore
            else:
                st.error("No documents loaded! Please load PDFs first.")
                st.stop()

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Rerun to show the new message
        st.rerun()

    # Process the last message if it's from the user
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        last_message = st.session_state.messages[-1]["content"]

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Search for relevant documents
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_kwargs={"k": 3}
                    )
                    relevant_docs = retriever.invoke(last_message)

                    # Create context from relevant documents
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])

                    # Simple response generation without complex LLM for now
                    response = f"Based on the documents, here's what I found:\n\n{context[:1000]}..."

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                    # Show sources
                    with st.expander("📖 Sources"):
                        for i, doc in enumerate(relevant_docs, 1):
                            st.markdown(f"**Source {i}:**")
                            st.markdown(doc.page_content[:300] + "...")
                            if i < len(relevant_docs):
                                st.markdown("---")

                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
