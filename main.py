import os
import sys
from dotenv import load_dotenv
from book_processor import BookProcessor
from vector_store import VectorStore
from sanderson_agents import SandersonAgents
from crewai import Crew

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY not found in environment variables.")
    print("Please add your OpenAI API key to the .env file.")
    sys.exit(1)

class SandersonRAG:
    def __init__(self):
        """initialize the sanderson rag system"""
        self.books_directory = "sanderson_books"
        self.book_processor = BookProcessor(self.books_directory)
        self.vector_store = VectorStore()
        self.agents = SandersonAgents(self.vector_store, self.book_processor, OPENAI_API_KEY)
        self.read_books = []
    
    def process_books(self):
        """process all books and add them to the vector store"""
        print("Processing books and adding them to the vector store...")
        
        # loop through all books and process them if needed
        for filename, metadata in self.book_processor.books_metadata.items():
            if not metadata['processed']:
                print(f"Processing {metadata['title']}...")
                chapters = self.book_processor.process_book(filename)
                self.vector_store.add_book_to_vectorstore(chapters, metadata['title'])
                print(f"Finished processing {metadata['title']}.")
        
        print("All books have been processed and added to the vector store.")
    
    def collect_reading_history(self):
        """collect the user's reading history"""
        print("\n--- Collecting Reading History ---\n")
        
        # first we need to know what they've read
        reading_agent = self.agents.create_reading_history_agent()
        reading_task = self.agents.create_reading_history_task(reading_agent)
        
        # TODO: añadir historial de lecturas x usuario?
        reading_crew = Crew(
            agents=[reading_agent],
            tasks=[reading_task],
            verbose=True)
        
        # run
        result = reading_crew.kickoff()
        
        # clean up the results - split by comma and strip whitespace
        self.read_books = [book.strip() for book in result.split(',')]
        
        print(f"\nUser has read: {', '.join(self.read_books)}\n")
    
    def answer_question(self, question):
        """answer a user's question without spoilers"""
        if not self.read_books:
            print("Error: Reading history not collected. Please run collect_reading_history() first.")
            return
        
        print(f"\n--- Answering Question: {question} ---\n")
        
        # two-step process: research, then respond
        research_agent = self.agents.create_research_agent()
        research_task = self.agents.create_research_task(research_agent, question, self.read_books)
        
        # first crew just does research
        research_crew = Crew(
            agents=[research_agent],
            tasks=[research_task],
            verbose=True)
        
        research_result = research_crew.kickoff()
        
        # now craft a nice response with the research
        response_agent = self.agents.create_response_agent()
        response_task = self.agents.create_response_task(
            response_agent, 
            question, 
            research_result, 
            self.read_books)
        
        response_crew = Crew(
            agents=[response_agent],
            tasks=[response_task],
            verbose=True)
        
        # get the final answer
        final_response = response_crew.kickoff()
        
        print("\n--- Final Response ---\n")
        print(final_response)
        
        return final_response

def main():
    """main function to run the sanderson rag system"""
    print("Welcome to the Brandon Sanderson RAG System!")
    print("This system will help you explore Brandon Sanderson's books without spoilers.")
    
    rag = SandersonRAG()
    
    # first run needs to process books - can take a while
    print("\nDo you want to process all books? This may take some time but is necessary for the first run.")
    process_books = input("Process books? (y/n): ").lower() == 'y'
    
    if process_books:
        rag.process_books()
    
    # find out what they've read
    rag.collect_reading_history()
    
    # keep answering questions until they quit
    while True:
        print("\nWhat would you like to know about Brandon Sanderson's books?")
        print("(Type 'exit' to quit)")
        
        question = input("Your question: ")
        
        if question.lower() == 'exit':
            print("Thank you for using the Brandon Sanderson RAG System. Goodbye!")
            break
        
        # sometimes the question is empty
        if not question.strip():
            print("Please enter a valid question!")
            continue
            
        rag.answer_question(question)

if __name__ == "__main__":
    main()

# Ideas para mejorar:
# - añadir una interfaz web con Gradio o Streamlit ???
# - guardar el historial de lectura en un archivo para no preguntar cada vez
