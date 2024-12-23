from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from tqdm import tqdm
import os

# Load environment variables
load_dotenv()

class PapersQAAgent:
    def __init__(self):
        # Load environment variables
        self.project_name = os.getenv('PROJECT_NAME')
        self.papers_dir = os.getenv('PAPERS_DIR')
        self.model_name = os.getenv('OLLAMA_MODEL')
        self.chroma_db_dir = os.getenv('CHROMA_DB_DIR')
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL')

        # Create papers directory if it doesn't exist
        os.makedirs(self.papers_dir, exist_ok=True)

        # Simple check for PDF files
        if not self._has_pdf_files():
            print(f"\n📚 Please add PDF papers to {self.papers_dir} directory")
            print("The agent will be ready once you add some papers.")
            return

        # Initialize LLM components
        try:
            self.llm = OllamaLLM(
                model=self.model_name,
                base_url=self.ollama_base_url
            )
            self.embeddings = OllamaEmbeddings(
                model=self.model_name,
                base_url=self.ollama_base_url
            )
            self.vector_store = None
            self.qa_chain = None
            os.makedirs(self.chroma_db_dir, exist_ok=True)
        except Exception:
            print("\n⚠️ Could not connect to Ollama server")
            print("Please make sure Ollama is running and try again")
            return
        
    def _has_pdf_files(self) -> bool:
        """Check if the papers directory contains any PDF files"""
        if not os.path.exists(self.papers_dir):
            return False
        
        pdf_files = [f for f in os.listdir(self.papers_dir) 
                     if f.lower().endswith('.pdf')]
        return len(pdf_files) > 0
        
    def load_papers(self):
        """Load and index all papers from the directory with progress tracking"""
        if not self._has_pdf_files():
            return f"No PDF files found in {self.papers_dir}. Please add some papers first."
        
        print("Starting papers loading process...")
        
        # Get total number of PDFs for progress tracking
        pdf_files = [f for f in os.listdir(self.papers_dir) if f.lower().endswith('.pdf')]
        print(f"Found {len(pdf_files)} PDF files to process")
        
        # Load PDFs from directory with progress bar
        loader = DirectoryLoader(
            self.papers_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        
        print("Loading documents...")
        documents = loader.load()
        
        if not documents:
            return f"No PDF files found in {self.papers_dir}"
        
        print(f"Splitting {len(documents)} documents into chunks...")
        # Split documents into chunks with progress tracking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        with tqdm(total=len(documents), desc="Splitting documents") as pbar:
            splits = text_splitter.split_documents(documents)
            pbar.update(len(documents))
        
        print(f"Creating vector store with {len(splits)} chunks...")
        # Create vector store with progress tracking
        with tqdm(total=len(splits), desc="Creating embeddings") as pbar:
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.chroma_db_dir
            )
            pbar.update(len(splits))
        
        print("Initializing QA chain...")
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=False,
            verbose=True
        )
        
        print("✅ Loading complete!")
        return f"Successfully loaded and indexed {len(documents)} documents with {len(splits)} total chunks"
        
    def search_papers(self, query: str) -> str:
        """Search through papers and return relevant information"""
        if not self.qa_chain:
            return "Please load papers first using load_papers()"
        
        try:
            # Use invoke with proper input format and extract only the result
            result = self.qa_chain.invoke({"query": query})
            return result["result"]
        except Exception as e:
            return f"Error searching papers: {str(e)}"
            
    def get_paper_metadata(self) -> str:
        """Return metadata about loaded papers"""
        if not self.vector_store:
            return "No papers loaded yet"
        
        try:
            collection = self.vector_store.get()
            return {
                "project_name": self.project_name,
                "papers_directory": self.papers_dir,
                "model_used": self.model_name,
                "chunks_indexed": len(collection['ids'])
            }
        except Exception as e:
            return f"Error getting metadata: {str(e)}"

