from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
import os
import shutil
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

DATA_PATH = 'data'
CHROMA_PATH = "chroma"

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob='*.md')
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    try:
        # Clear out the database first.
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)

        # Create a new DB from the documents.
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma.from_documents(
            chunks, embeddings, persist_directory=CHROMA_PATH
        )
        db.persist()
        print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    except Exception as e:
        print(f"An error occurred while saving to Chroma: {e}")

if __name__ == "__main__":
    try:
        documents = load_documents()
        print(f"Loaded {len(documents)} documents.")
        chunks = split_text(documents)
        save_to_chroma(chunks)
        for chunk in chunks:
            print(chunk.page_content)
            print(chunk.metadata)
    except Exception as e:
        print(f"An error occurred: {e}")