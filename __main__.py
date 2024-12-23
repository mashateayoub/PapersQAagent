import sys
from dotenv import load_dotenv
from PapersQAAgent import PapersQAAgent
import os

# Load environment variables
load_dotenv()

def chat_loop(agent):
    """Handle the chat interaction loop"""
    print("\nYou can start asking questions about the papers.")
    print("Type 'exit' to quit, 'help' for commands.")
    
    while True:
        try:
            query = input("\nQuestion: ").strip()
            
            if query.lower() == 'exit':
                print("Goodbye!")
                sys.exit(0)
                
            elif query.lower() == 'help':
                print("\nAvailable commands:")
                print("- exit: Quit the program")
                print("- help: Show this help message")
                print("- metadata: Show information about loaded papers")
                continue
                
            elif query.lower() == 'metadata':
                print(agent.get_paper_metadata())
                continue
                
            if not query:
                continue
                
            response = agent.search_papers(query)
            print("\nResponse:", response)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"\nError: {str(e)}")

def main():
    # Initialize the papers QA agent
    agent = PapersQAAgent()  # Create an instance of the class
    
    # Create papers directory if it doesn't exist
    if not os.path.exists(agent.papers_dir):
        os.makedirs(agent.papers_dir, exist_ok=True)
        print("Created papers directory. Please add PDF papers to ./papers/")
        return
    agent.load_papers()  # Now this will work correctly
    # Start the chat loop
    chat_loop(agent)

if __name__ == "__main__":
    main()
