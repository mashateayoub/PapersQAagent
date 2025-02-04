import sys
from dotenv import load_dotenv
from PapersQAAgent import PapersQAAgent
import os
import warnings


# Filter out deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
                print("- audio: Start real-time audio input")
                continue
                
            elif query.lower() == 'metadata':
                print(agent.get_paper_metadata())
                continue

            elif query.lower() == 'audio':
                query = agent.listen_and_convert()
                
            if not query:
                continue
                
            response = agent.search_papers(query)
            print("\nResponse:", response)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"\nError: {str(e)}")

def print_help():
    print("\nAvailable commands:")
    print("- exit or `CTRL-C`: Quit the program")
    print("- help: Show this help message")
    print("- metadata: Show information about loaded papers")
    print("- audio: Start real-time audio input")

def main():
    # Initialize the agent
    agent = PapersQAAgent()
    print_help()

    while True:
        try:
            # ANSI escape codes for colors
            YELLOW = '\033[93m'
            RESET = '\033[0m'
            
            # Get input and immediately clear the input line
            command = input("Question: ")
            sys.stdout.write('\033[F\033[K')  # Move cursor up and clear line
            
            if command.lower() == 'exit':
                break
            elif command.lower() == 'help':
                print_help()
            elif command.lower() == 'audio':
                text = agent.listen_and_convert()
                print(f"{YELLOW}Question: {text}{RESET}")
                print(agent.search_papers(text))
            else:
                print(f"{YELLOW}Question: {command}{RESET}")
                print(agent.search_papers(command))
                
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            sys.exit(0)
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye! 👋")
        sys.exit(0)