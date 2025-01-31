import os
import hashlib
import json
from tqdm import tqdm
from dotenv import load_dotenv
import speech_recognition as sr
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import time
import sys
import threading
from itertools import cycle

class PapersQAAgent:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.project_name = os.getenv('PROJECT_NAME')
        self.papers_dir = os.getenv('PAPERS_DIR')
        self.model_name = os.getenv('OLLAMA_MODEL')
        self.chroma_db_dir = os.getenv('CHROMA_DB_DIR')
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL')
        self.hashes_file = os.path.join(self.chroma_db_dir, "file_hashes.json")
        self.reasoning = False  # Add this line

        # Create papers directory if it doesn't exist
        os.makedirs(self.papers_dir, exist_ok=True)
        os.makedirs(self.chroma_db_dir, exist_ok=True)

        # Initialize components
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
        except Exception:
            print("\n⚠️ Could not connect to Ollama server")
            print("Please make sure Ollama is running and try again")
            return

        # Check for PDF files and load papers if available
        if self._has_pdf_files():
            self.load_papers()
        else:
            print(f"\n📚 Please add PDF papers to {self.papers_dir} directory")
            print("The agent will be ready once you add some papers.")

    def _has_pdf_files(self) -> bool:
        """Check if the papers directory contains any PDF files"""
        if not os.path.exists(self.papers_dir):
            return False

        pdf_files = [f for f in os.listdir(self.papers_dir) 
                     if f.lower().endswith('.pdf')]
        return len(pdf_files) > 0

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate the hash of a file"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def _load_hashes(self) -> dict:
        """Load stored file hashes from the JSON file"""
        if os.path.exists(self.hashes_file):
            with open(self.hashes_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_hashes(self, hashes: dict):
        """Save file hashes to the JSON file"""
        with open(self.hashes_file, 'w') as f:
            json.dump(hashes, f)

    def load_papers(self):
        """Load and index all papers from the directory with progress tracking"""
        if not self._has_pdf_files():
            return f"No PDF files found in {self.papers_dir}. Please add some papers first."

        print("Starting papers loading process...")

        # Get total number of PDFs for progress tracking
        pdf_files = [f for f in os.listdir(self.papers_dir) if f.lower().endswith('.pdf')]
        print(f"Found {len(pdf_files)} PDF files to process")

        # Load existing hashes
        stored_hashes = self._load_hashes()
        current_hashes = {f: self._calculate_file_hash(os.path.join(self.papers_dir, f)) for f in pdf_files}

        # Check if any files have changed
        files_to_process = [f for f in pdf_files if f not in stored_hashes or stored_hashes[f] != current_hashes[f]]

        if not files_to_process:
            print("No changes detected in PDF files. Skipping loading process.")
            self._initialize_qa_chain()
            return "No changes detected. Papers are already loaded and indexed."

        print(f"Processing {len(files_to_process)} files...")

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

        # Save the current hashes
        self._save_hashes(current_hashes)

        print("✅ Loading complete!")
        return f"Successfully loaded and indexed {len(documents)} documents with {len(splits)} total chunks"

    def _initialize_qa_chain(self):
        """Initialize the QA chain if not already initialized"""
        if self.vector_store is None:
            self.vector_store = Chroma(
                persist_directory=self.chroma_db_dir,
                embedding_function=self.embeddings  # Changed from embedding to embedding_function
            )
        if self.qa_chain is None:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": 3}
                ),
                return_source_documents=False,
                verbose=True
            )


    def _animate_reasoning(self):
        """Display an animation while the model is reasoning"""
        # ANSI escape codes for colors
        YELLOW = '\033[93m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        
        frames = ["reasonning...", "rea ", "reason", "reasonning..."]
        for frame in cycle(frames):
            if not self.reasoning:
                break
            sys.stdout.write('\r' + frame)
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\r' + ' ' * 50 + '\r')
        sys.stdout.flush()

    def _format_elapsed_time(self, elapsed_ms: int) -> str:
        """Format elapsed time in human-readable format"""
        if elapsed_ms < 1000:  # Less than 1 second
            return f"{elapsed_ms}ms"
        elif elapsed_ms < 60000:  # Less than 1 minute
            seconds = elapsed_ms / 1000
            return f"{seconds:.1f}s"
        else:  # 1 minute or more
            minutes = elapsed_ms / (1000 * 60)
            return f"{minutes:.1f}min"

    def search_papers(self, query: str) -> str:
        """Search through papers and return relevant information"""
        if not self.qa_chain:
            return "Please load papers first using load_papers()"

        # ANSI escape codes for colors
        YELLOW = '\033[93m'
        GREEN = '\033[92m'
        RESET = '\033[0m'

        try:
            # Start timing
            start_time = time.time()

            # Start reasoning animation
            self.reasoning = True
            animation_thread = threading.Thread(target=self._animate_reasoning)
            animation_thread.start()

            # Capture stdout to hide chain output
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

            # Use invoke with proper input format and extract only the result
            result = self.qa_chain.invoke({"query": query})
            
            # Calculate elapsed time
            elapsed_ms = int((time.time() - start_time) * 1000)  # Convert to milliseconds
            formatted_time = self._format_elapsed_time(elapsed_ms)
            
            # Restore stdout
            sys.stdout = original_stdout

            # Stop reasoning animation and clear it
            self.reasoning = False
            animation_thread.join()
            print('\r' + ' ' * 50 + '\r', end='')

            # Process the response and return it with colors and timing
            response = self._process_think_output(result["result"])
            return f"{GREEN}Response: ({formatted_time}) {response}{RESET}"
        except Exception as e:
            self.reasoning = False
            sys.stdout = sys.__stdout__
            return f"Error searching papers: {str(e)}"

    def _process_think_output(self, response: str) -> str:
        """Process the <think> output from the reasoning model"""
        think_start = response.find("<think>")
        think_end = response.find("</think>")
        if think_start != -1 and think_end != -1:
            # Skip displaying the reasoning, just return the response
            return response[think_end + len('</think>'):].strip()
        return response.strip()

    def get_paper_metadata(self) -> str:
        """Return metadata about loaded papers in a formatted string"""
        if not self.vector_store:
            return "No papers loaded yet"

        try:
            collection = self.vector_store.get()
            pdf_files = [f for f in os.listdir(self.papers_dir) if f.lower().endswith('.pdf')]

            metadata = [
                "\n📊 Papers QA System Metadata",
                "=" * 30,
                f"🔹 Project Name: {self.project_name}",
                f"🔹 Model: {self.model_name}",
                f"🔹 Papers Location: {self.papers_dir}",
                f"🔹 Number of Papers: {len(pdf_files)}",
                f"🔹 Total Chunks Indexed: {len(collection['ids'])}",
                f"🔹 Vector DB Location: {self.chroma_db_dir}",
                "\n📑 Loaded Papers:",
                "-" * 20
            ]

            # Add list of papers with sizes
            for pdf in pdf_files:
                file_path = os.path.join(self.papers_dir, pdf)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                metadata.append(f"📄 {pdf} ({size_mb:.2f} MB)")

            return "\n".join(metadata)
        except Exception as e:
            return f"Error getting metadata: {str(e)}"

    def listen_and_convert(self) -> str:
        """Listen to real-time audio and convert it to text using SpeechRecognition"""
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            # Show "Listening..." message
            sys.stdout.write('\r' + ' ' * 50 + '\r')  # Clear line
            print("Listening for your question...")
            
            try:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=20)
                
                # Show "Recognized Text" message
                sys.stdout.write('\r' + ' ' * 50 + '\r')  # Clear previous line
                text = recognizer.recognize_google(audio)
                print(f"Recognized Text: {text}")
                
                # Clear both messages after recognition
                sys.stdout.write('\033[F\033[K')  # Move up and clear line
                sys.stdout.write('\033[F\033[K')  # Move up and clear line again
                
                return text
                
            except sr.UnknownValueError:
                sys.stdout.write('\r' + ' ' * 50 + '\r')  # Clear line
                return "Sorry, I could not understand the audio."
            except sr.RequestError as e:
                sys.stdout.write('\r' + ' ' * 50 + '\r')  # Clear line
                return f"Could not request results; {e}"
            except Exception as e:
                sys.stdout.write('\r' + ' ' * 50 + '\r')  # Clear line
                return f"Error: {str(e)}"