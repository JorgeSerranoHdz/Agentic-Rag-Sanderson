# Brandon Sanderson RAG System

A simple RAG system for Brandon Sanderson's books that avoids spoilers based on what you've read.

## Features

- Processes EPUB files of Brandon Sanderson's books
- Asks which books you've read to avoid spoilers
- Answers questions based only on the books you've read
- Uses CrewAI for specialized agents

## Requirements

- Python 3.8+
- OpenAI API key

## Installation

1. Clone this repository
2. Install the required packages:
```
pip install -r requirements.txt
```
3. Add your OpenAI API key to the `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

1. Put your Brandon Sanderson EPUB books in the `sanderson_books` directory
2. Run the main script:
```
python main.py
```
3. The system will ask if you want to process all books (necessary for first run)
4. Tell the system which books you've read
5. Ask questions about the books without getting spoilers

## How It Works

1. **Book Processing**: Extracts text from EPUB files and splits it into chunks
2. **Vector Storage**: Stores book content for efficient retrieval
3. **Reading History**: Asks which books you've read to avoid spoilers
4. **Question Answering**: Uses agents to research and craft spoiler-free responses

## Project Structure

- `main.py`: Main script that ties everything together
- `book_processor.py`: Handles extracting text from EPUB files
- `vector_store.py`: Manages the vector database
- `sanderson_agents.py`: Defines the CrewAI agents and tasks

## Limitations

- The system relies on accurate book metadata for spoiler prevention.
- Processing large books can take significant time and memory.
- The quality of answers depends on the OpenAI model used.

## Future Improvements

- Add support for more book formats (PDF, MOBI, etc.)
- Improve parsing of reading history responses
- Add a web interface for easier interaction
- Implement more sophisticated spoiler detection

## License

This project is licensed under the MIT License - see the LICENSE file for details. 