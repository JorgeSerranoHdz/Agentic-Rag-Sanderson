import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

# FIXME: a veces da este error, pero parece que funciona igual:
# 2025-02-28 21:19:13,646 - 25180 - base.py-base:277 - WARNING: Warning: model not found. Using cl100k_base encoding.

class VectorStore:
    def __init__(self, persist_directory="./chroma_db"):
        """initialize the vector store"""
        self.persist_directory = persist_directory
        # using text-embedding-3-large for better semantic search quality
        # costs more but gives much better results than ada or the small version
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # check if directory exists, if not create it
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            self.db = None
        else:
            # try loading the db
            try:
                self.db = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)
            except:
                self.db = None
    
    def add_book_to_vectorstore(self, chapters, book_title):
        """add book chapters to the vector store"""
        # setup text splitter with decent defaults
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len)
        
        documents = []
        print(f"Processing chunks for {book_title}...")
        
        for chapter in chapters:
            # chunk it up
            chapter_chunks = text_splitter.split_text(chapter['content'])
            
            # make docs from chunks
            for i, chunk in enumerate(chapter_chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={"book": book_title,
                              "chunk_id": i,
                              "source": chapter['filename']
                    })
                documents.append(doc)
        
        # db handling - create or update
        if self.db is None:
            self.db = Chroma.from_documents(documents=documents,
                                           embedding=self.embeddings,
                                           persist_directory=self.persist_directory)
        else:
            self.db.add_documents(documents)
        
        # save it
        self.db.persist()
        
        print(f"Added {len(documents)} chunks from {book_title} to vector store.")
    
    def search(self, query, book_titles=None, k=5):
        """search for relevant documents in the vector store"""
        if self.db is None:
            print("No vector store available. Please add books first.")
            return []
        
        # filter by books if needed
        filter_dict = None
        if book_titles:
            filter_dict = {"book": {"$in": book_titles}}
        
        # do the search
        results = self.db.similarity_search(query,
                                           k=k,
                                           filter=filter_dict)
        
        return results

# Note to self: 
# revisar tamaño de los chunks
# revisar coste de la API con el modelo large

# Approximate costs for 3M tokens:
# - text-embedding-3-large: $0.39 (3M * $0.13/1M)
# - text-embedding-3-small: $0.06 (3M * $0.02/1M)
# The large model processes ~350-450k tokens/min
# Estimated time for 3M tokens: ~10-15 minutes total
