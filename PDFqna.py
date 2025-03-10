import streamlit as st
import os
import time
import shutil
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv

# Page configuration
st.set_page_config(
    page_title="PDF Question Answering",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        width: 100%;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .source-box {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 5px solid #4CAF50;
    }
    .chat-user {
        background-color: #e1f5fe;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .chat-ai {
        background-color: #f0f4c3;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
if "document_source" not in st.session_state:
    st.session_state.document_source = None

# Create LLM
@st.cache_resource(show_spinner=False)
def get_llm(model_name):
    return ChatGroq(groq_api_key=groq_api_key, model_name=model_name)

# Main processing functions
def create_vector_embeddings(files_directory, embedding_model, chunk_size, chunk_overlap):
    with st.spinner("Processing documents and creating embeddings..."):
        try:
            # Initialize embedding model
            st.session_state.embeddings = OllamaEmbeddings(model=embedding_model)
            
            # Load documents
            st.session_state.loader = PyPDFDirectoryLoader(files_directory)
            st.session_state.docs = st.session_state.loader.load()
            
            if not st.session_state.docs:
                st.error(f"No PDF documents found in directory: {files_directory}")
                return False
            
            # Split documents
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                st.session_state.docs[:50]
            )
            
            # Create vector store
            # Use a unique persist directory based on the source
            persist_directory = f"chroma_db_{hash(files_directory)}"
            
            # Delete existing DB if it exists
            if os.path.exists(persist_directory):
                shutil.rmtree(persist_directory)
            
            st.session_state.vectors = Chroma.from_documents(
                documents=st.session_state.final_documents,
                embedding=st.session_state.embeddings,
                persist_directory=persist_directory
            )
            
            # Store document source for reference
            st.session_state.document_source = files_directory
            
            return True
        
        except Exception as e:
            st.error(f"Error creating embeddings: {str(e)}")
            return False

def process_query(query, llm_model, k_docs):
    if "vectors" not in st.session_state:
        st.error("Please process documents first before asking questions.")
        return None
    
    try:
        llm = get_llm(llm_model)
        
        # Create prompt template with chat history
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the question based ONLY on the following context. If you don't know the answer based on the context, say "I don't have enough information to answer this question."
            
            <context>
            {context}
            </context>
            
            Chat History:
            {chat_history}
            
            Question: {input}
            
            Provide a concise, helpful response.
            """
        )
        
        # Create document chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Create advanced retriever
        retriever = st.session_state.vectors.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": k_docs,
                "fetch_k": k_docs * 2,
                "lambda_mult": 0.5  # Diversity parameter
            }
        )
        
        # Timers for performance metrics
        start_retrieval = time.process_time()
        retrieved_docs = retriever.get_relevant_documents(query)
        retrieval_time = time.process_time() - start_retrieval
        
        start_generation = time.process_time()
        answer = document_chain.invoke({
            "input": query,
            "context": retrieved_docs,
            "chat_history": st.session_state.memory.load_memory_variables({})["chat_history"]
        })
        generation_time = time.process_time() - start_generation
        
        # Update memory
        st.session_state.memory.chat_memory.add_user_message(query)
        st.session_state.memory.chat_memory.add_ai_message(answer)
        
        return {
            "answer": answer,
            "docs": retrieved_docs,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": retrieval_time + generation_time
        }
    
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None

# Sidebar configuration with minimal options
with st.sidebar:
    st.title("📚 Document QA")
    
    # Model options in an expander to declutter
    with st.expander("Model Settings"):
        llm_model = st.selectbox(
            "LLM Model", 
            ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
            index=0
        )
        
        embedding_model = st.selectbox(
            "Embedding Model", 
            ["llama3.2:latest"],
            index=0
        )
        
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50)
        
        k_docs = st.slider("Documents to retrieve", 3, 10, 5)
    
    # Document status with cleaner display
    if "docs" in st.session_state:
        st.success("Documents loaded successfully")
        if st.session_state.get("document_source"):
            st.info(f"Document source: {os.path.basename(st.session_state.document_source)}")
        else:
            st.info("No document source provided.")

        st.info(f"Pages: {len(st.session_state.docs)}")
        st.info(f"Chunks: {len(st.session_state.final_documents)}")
    
    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.success("Conversation cleared!")

# Main content area
st.title("PDF Question Answering")
st.markdown("Upload PDFs and ask questions about their content.")

# File uploader with clear visual separation
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Process uploads or use default directory - clear distinction between options
col1, col2 = st.columns(2)

with col1:
    if uploaded_files:
        if st.button("Process Uploaded PDFs", key="process_uploaded"):
            # Save uploaded files to temporary directory
            temp_dir = "temp_pdfs"
            # Ensure directory exists and is empty
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            
            # Save new files
            for file in uploaded_files:
                with open(os.path.join(temp_dir, file.name), "wb") as f:
                    f.write(file.getbuffer())
            
            success = create_vector_embeddings(
                temp_dir, 
                embedding_model, 
                chunk_size, 
                chunk_overlap
            )
            
            if success:
                st.success(f"Successfully processed {len(uploaded_files)} PDF document(s).")

with col2:
    if st.button("Use Default Documents", key="use_default"):
        default_dir = "details"
        if not os.path.exists(default_dir):
            os.makedirs(default_dir)
            st.warning(f"Default directory '{default_dir}' was created. Please add PDFs there.")
        else:
            success = create_vector_embeddings(
                default_dir, 
                embedding_model, 
                chunk_size, 
                chunk_overlap
            )
            if success:
                st.success("Default documents processed successfully.")

st.markdown("---")

if st.session_state.chat_history:
    st.subheader("Conversation")
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:  # User message
                st.markdown(f"""<div class="chat-user"><strong>You:</strong> {message}</div>""", unsafe_allow_html=True)
            else:  # AI message
                st.markdown(f"""<div class="chat-ai"><strong>AI:</strong> {message}</div>""", unsafe_allow_html=True)

st.subheader("Ask a Question")
user_query = st.text_input("Enter your question about the documents:", key="query_input")
query_button = st.button("Ask", key="submit_query")

if query_button and user_query:
    if "vectors" not in st.session_state:
        st.error("Please process documents first before asking questions.")
    else:
        st.session_state.chat_history.append(user_query)
        
        with st.spinner("Searching documents for an answer..."):
            result = process_query(user_query, llm_model, k_docs)
        
        if result:
            st.session_state.chat_history.append(result["answer"])
            
            st.rerun()

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if user_query and "vectors" in st.session_state and not query_button:
    if len(st.session_state.chat_history) >= 2:
        st.subheader("Answer")
        st.markdown(st.session_state.chat_history[-1])
        
        if st.session_state.last_result:
            with st.expander("View Source Documents", expanded=False):
                for i, doc in enumerate(st.session_state.last_result["docs"]):
                    source = doc.metadata.get('source', 'Unknown')
                    page = doc.metadata.get('page', 'Unknown')
                    
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>Source {i+1}:</strong> {os.path.basename(source)} - Page {page}<br>
                        <div style="margin-top:10px;">{doc.page_content[:300]}...</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                if st.button("👍 Helpful"):
                    st.success("Thanks for your feedback!")
            with col2:
                if st.button("👌 Somewhat Helpful"):
                    st.success("Thanks for your feedback!")
            with col3:
                if st.button("👎 Not Helpful"):
                    st.success("Thanks for your feedback!")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Powered by LangChain & Groq</p>", unsafe_allow_html=True)