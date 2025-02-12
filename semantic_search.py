import os
import argparse
import asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

def valid_file_path(file_path: str) -> str:
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File not found: {abs_path}")
    if not os.path.isfile(abs_path):
        raise ValueError(f"Invalid type path: {abs_path}")
    return abs_path

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--question", "-q", type=str, required=True, help="The question to ask")
    parser.add_argument("--document", "-d", type=valid_file_path, required=True, help="The document to load and search in")
    parser.add_argument("--model", "-m", type=str, default="llama3.2", choices=['llama3.2', 'llama3', 'mistral', 'deepseek-r1'], help="The model to use (need to be available in Ollama)")
    args = parser.parse_args()

    if args.debug:
        print(f"Using question: {args.question}")
        print(f"Using document: {args.document}")
        print(f"Using model: {args.model}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=10, add_start_index=True
    )

    embeddings = OllamaEmbeddings(model=args.model)
    vector_store = InMemoryVectorStore(embeddings)  # HINT: can use PGVector

    doc_loader = PyPDFLoader(args.document)
    loaded_docs = doc_loader.load()
    doc_splits = text_splitter.split_documents(loaded_docs)
    ids = vector_store.add_documents(documents=doc_splits)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1}
    )

    results = await retriever.ainvoke(args.question)
    print(results)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exiting...")
