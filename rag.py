import argparse
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

def load_resources():
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-header")))
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    return all_splits

class State(TypedDict):
    question: str
    context: List[Document]
    answer:str

def make_retrieve(vs: VectorStore):
    def retrieve(state: State):
        retrieved_docs = vs.similarity_search(state["question"])
        return { "context": retrieved_docs }
    return retrieve

def make_generate(llm: BaseChatModel, prompt):
    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({ "question": state["question"], "context": docs_content })
        response = llm.invoke(messages)
        return { "answer": response }
    return generate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="llama3.2", choices=["mistral-nemo", "llama3.2", "gpt-4o-mini"])
    parser.add_argument("--provider", "-p", type=str, default="ollama", choices=["ollama", "openai"])
    args = parser.parse_args()

    llm = init_chat_model(model=args.model, model_provider=args.provider)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large") if args.provider == "openai" else OllamaEmbeddings(model=args.model)
    vector_store = InMemoryVectorStore(embeddings)

    prompt = hub.pull("rlm/rag-prompt")

    docs_splits_all = load_resources()
    _ = vector_store.add_documents(docs_splits_all)
    graph_builder = StateGraph(State).add_sequence([make_retrieve(vector_store), make_generate(llm, prompt)])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    response = graph.invoke({ "question": "What is Task Decomposition?" })
    print(response["answer"])


if __name__ == "__main__":
    main()
