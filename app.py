import streamlit as st
from typing import List, Dict
from pathlib import Path
import os
from dotenv import load_dotenv
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
from PyPDF2 import PdfReader
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK data at startup
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ResearchAssistant:
    def __init__(self):
        # Initialize with HuggingFace Hub
        load_dotenv()
        self.llm = self._initialize_llm()
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
    def _initialize_llm(self):
        # Use HuggingFace Hub for hosted inference
        return HuggingFaceHub(
            repo_id="google/flan-t5-large",  # Using a smaller model for Streamlit Cloud
            model_kwargs={"temperature": 0.7, "max_length": 512},
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_TOKEN")
        )
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        if pdf_file.name.endswith('.pdf'):
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        else:
            raise ValueError("Please upload a PDF file")
    
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms using TF-IDF"""
        # Tokenize and remove stopwords
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text.lower())
        filtered_tokens = [w for w in word_tokens if w.isalnum() and w not in stop_words]
        
        # Use TF-IDF to identify important terms
        vectorizer = TfidfVectorizer(max_features=20)
        tfidf_matrix = vectorizer.fit_transform([' '.join(filtered_tokens)])
        
        # Get top terms
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        key_terms = [feature_names[i] for i in scores.argsort()[-10:][::-1]]
        return key_terms
    
    def setup_rag_pipeline(self, text: str):
        """Set up RAG pipeline with document chunks and FAISS index"""
        # Split text into chunks
        documents = self.text_splitter.create_documents([text])
        
        # Create FAISS index
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        # Create conversational chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(),
            memory=self.memory,
            return_source_documents=True
        )
    
    def ask_question(self, question: str) -> Dict:
        """Ask a question about the research paper"""
        if not hasattr(self, 'qa_chain'):
            raise ValueError("Please upload a research paper first")
        
        response = self.qa_chain({"question": question})
        return {
            "answer": response["answer"],
            "sources": [doc.page_content for doc in response["source_documents"]]
        }

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Research Paper Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 AI Research Assistant")
    st.write("Upload a research paper and ask questions about it!")
    
    # Initialize Research Assistant
    if 'assistant' not in st.session_state:
        st.session_state.assistant = ResearchAssistant()
    
    # File upload
    uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type="pdf")
    
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            # Extract text
            text = st.session_state.assistant.extract_text_from_pdf(uploaded_file)
            
            # Extract and display key terms
            key_terms = st.session_state.assistant.extract_key_terms(text)
            st.subheader("📝 Key Terms")
            st.write(", ".join(key_terms))
            
            # Setup RAG pipeline
            st.session_state.assistant.setup_rag_pipeline(text)
            st.success("✅ Paper processed successfully!")
    
    # Question input
    st.subheader("💭 Ask Questions")
    question = st.text_input("What would you like to know about the paper?")
    if question:
        try:
            with st.spinner("Analyzing..."):
                response = st.session_state.assistant.ask_question(question)
                
                st.subheader("🤖 Answer")
                st.write(response["answer"])
                
                with st.expander("📑 View Sources"):
                    for i, source in enumerate(response["sources"], 1):
                        st.write(f"Source {i}:")
                        st.write(source)
                        st.write("---")
        except ValueError as e:
            st.error(str(e))
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Made with ❤️ using LangChain, HuggingFace, and Streamlit"
    )

if __name__ == "__main__":
    main()