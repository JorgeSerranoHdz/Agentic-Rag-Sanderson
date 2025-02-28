import os
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re

class BookProcessor:
    def __init__(self, books_directory):
        """initialize the processor with the books directory"""
        self.books_directory = books_directory
        self.books_metadata = {}
        self._load_books()
    
    def _load_books(self):
        """load metadata for all books in the directory"""
        for filename in os.listdir(self.books_directory):
            if filename.endswith('.epub'):
                book_path = os.path.join(self.books_directory, filename)
                try:
                    book = epub.read_epub(book_path)
                    title = book.get_metadata('DC', 'title')
                    # fallback to filename if no title found
                    title = title[0][0] if title else filename.replace('.epub', '')
                    
                    # TODO: añadir más metadatos como fecha, etc
                    self.books_metadata[filename] = {
                        'title': title,
                        'path': book_path,
                        'processed': False,}
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    
    def process_book(self, filename):
        """process a book and extract its text content"""
        if filename not in self.books_metadata:
            print(f"Book {filename} not found.")
            return []
        
        book_path = self.books_metadata[filename]['path']
        book = epub.read_epub(book_path)
        chapters = []
        
        # grab all the html docs from the epub
        items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        
        print(f"Processing {filename}...") 
        for item in items:
            # don't need these parts
            if 'cover' in item.get_name().lower() or 'toc' in item.get_name().lower():
                continue
                
            # extract the actual text from html
            soup = BeautifulSoup(item.get_body_content(), 'html.parser')
            text = soup.get_text()
            # clean up whitespace - regex to the rescue
            text = re.sub(r'\s+', ' ', text).strip()
            
            if text:  # empty chapters are useless
                chapters.append({
                    'content': text,
                    'filename': filename,
                    'book_title': self.books_metadata[filename]['title']
                })
        
        self.books_metadata[filename]['processed'] = True
        return chapters
    
    def get_all_books(self):
        """get a list of all available books"""
        # just extract the titles from metadata
        return [metadata['title'] for _, metadata in self.books_metadata.items()]
    
    def get_filename_by_title(self, title):
        """get the filename for a book by its title"""
        # case insensitive search
        for filename, metadata in self.books_metadata.items():
            if metadata['title'].lower() == title.lower():
                return filename
        return None 