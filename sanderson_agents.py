from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from typing import List

# TODO: hay un bug raro con crewai cuando se ejecuta en Windows
# a veces se queda colgado esperando respuesta, hay que reiniciar

class SandersonAgents:
    def __init__(self, vector_store, book_processor, openai_api_key):
        """initialize the sanderson agents"""
        self.vector_store = vector_store
        self.book_processor = book_processor
        self.openai_api_key = openai_api_key
        self.read_books = []  # keep track of what they've read
        
        # setup the LLM - gpt4o mini is good enough and cheaper
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,  # bit of creativity but not too wild
            api_key=openai_api_key,
            # Disable token counting to avoid the tokenizer error
            max_tokens=None,
            streaming=True)
    
    def create_reading_history_agent(self):
        """create agent to collect user's reading history"""
        return Agent(role="Reading History Collector",
                    goal="Collect which Brandon Sanderson books the user has read",
                    backstory="You are an expert on Brandon Sanderson's works. Ask users which books they've read to avoid spoilers.",
                    verbose=True,
                    llm=self.llm,
                    tools=[Tool(name="get_all_books",
                               func=self.book_processor.get_all_books,
                               description="Get list of all available Brandon Sanderson books")])
    
    def create_research_agent(self):
        """create agent to research information from books"""
        # this is the workhorse that digs through the books
        return Agent(role="Cosmere Scholar",
                    goal="Research information without revealing spoilers",
                    backstory="You are a scholar of the Cosmere. You only provide information from books the user has read.",
                    verbose=True,
                    llm=self.llm,
                    tools=[Tool(name="search_books",
                               func=self._search_books,
                               description="Search for information in books the user has read")])
    
    def create_response_agent(self):
        """create agent to craft final responses"""
        # no tools needed for this one, just crafting responses
        return Agent(role="Cosmere Guide",
                    goal="Provide spoiler-free responses about Brandon Sanderson's books",
                    backstory="You craft responses that help readers understand books they've read without spoiling future events.",
                    verbose=True,
                    llm=self.llm)
    
    def _search_books(self, query):
        """search for information in books the user has read"""
        results = self.vector_store.search(query, book_titles=self.read_books, k=5)
        
        if not results:
            return "No relevant information found in the books you've read."
        
        # format nicely for the agent to use
        formatted_results = ""
        for i, doc in enumerate(results):
            formatted_results += f"Source {i+1} (from {doc.metadata['book']}):\n{doc.page_content}\n\n"
        
        return formatted_results
    
    def create_reading_history_task(self, agent):
        """create task to collect reading history"""
        # Get the list of books first to provide it in the context
        all_books = self.book_processor.get_all_books()
        
        return Task(
            description="Ask which Brandon Sanderson books the user has read. Present the list of available books and collect responses.",
            agent=agent,
            context=[{
                "description": "List of available Brandon Sanderson books",
                "expected_output": "A list of book titles the user has read",
                "available_books": all_books
            }],
            expected_output="A list of book titles the user has read")
    
    def create_research_task(self, agent, question, read_books):
        """create task to research information for a question"""
        # gotta update this every time or it gets confused
        self.read_books = read_books
        
        return Task(
            description=f"Research information to answer: \"{question}\". Only use information from books they've read: {', '.join(read_books)}",
            agent=agent,
            context=[{
                "description": f"The user has read the following books: {', '.join(read_books)}",
                "expected_output": "Relevant information from the books",
                "read_books": read_books,
                "question": question
            }],
            expected_output="Relevant information from the books")
    
    def create_response_task(self, agent, question, research_results, read_books):
        """create task to craft final response"""
        # this is where the magic happens - crafting spoiler-free responses
        return Task(
            description=f"Craft a response to: \"{question}\". Use the research results but avoid spoilers. User has read: {', '.join(read_books)}",
            agent=agent,
            context=[{
                "description": f"Research results for question: \"{question}\"",
                "expected_output": "A helpful, spoiler-free response",
                "research_results": research_results,
                "read_books": read_books,
                "question": question
            }],
            expected_output="A helpful, spoiler-free response") 
