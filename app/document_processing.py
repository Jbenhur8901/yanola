import logging
from PyPDF2 import PdfReader
from langchain_community.document_loaders import DirectoryLoader, SeleniumURLLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_voyageai import VoyageAIEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Voyage AI embeddings
embedding = VoyageAIEmbeddings(model="voyage-large-2")

def process_pdf_directory(path):
    """
    Process PDF files from a directory.

    Args:
    - path (str): Path to the directory containing PDF files.

    Returns:
    - List of text chunks extracted from the PDF files.
    """
    chunks = []
    try:
        loader = DirectoryLoader(path, glob="./*.pdf", loader_cls=PyPDFLoader)
        files = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300, separators="\n\n")
        for file in files:
            chunks.extend(text_splitter.split_documents([file]))
    except Exception as e:
        logger.error(f"Error processing PDF directory: {e}")
    return chunks


def url_loaders(urls, index):
    """
    Load documents from URLs, extract text content, and add them to a vector store for indexing.

    Args:
    - urls (list): List of URLs.
    - index (str): Name of the vector store index.

    Returns:
    - Success message or error message.
    """
    try:
        vector = PineconeVectorStore(index_name=index, embedding=embedding)
        Loader = SeleniumURLLoader(urls=[urls])
        documents = Loader.load()
        for doc in documents:
            if doc.page_content != "":
                subject_matter = doc.metadata["title"]
                source = doc.metadata["source"]
                splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300, separators="\n\n")
                chunks = splitter.split_documents([doc])
                ids = vector.add_documents(chunks)
                return f"{subject_matter} qui a pour source de l'url: {source}."
            else:
                return "No content found in the documents."
    except Exception as e:
        logger.error(f"Error loading documents from URLs: {e}")
        return f"Error loading documents from URLs: {e}"

def delete(ids,index):
    vectordb = PineconeVectorStore(index_name=index,
                                   embedding=embedding)
    vectordb.delete(ids=ids,delete_all=True)
    return "Success"