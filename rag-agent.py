import argparse
import asyncio
import os
import collect_urls
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from typing_extensions import List, TypedDict


async def load_resources(url, max_pages: int, debug=False):
    if url is None:
        return []
    urls = await collect_urls.afrom_website(
        start_url=url,
        url_limit=max_pages,
        concurrency=10,
        pattern=None,
        debug=debug
    )

    if debug:
        print(f"Loading resources from {len(urls)} urls on domain {url}")
    loader = WebBaseLoader(urls)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    return all_splits

class State(TypedDict):
    question: str
    context: List[Document]
    answer:str

async def make_retrieve(vs: VectorStore):
    @tool(response_format="content_and_artifact")
    async def retrieve(state: State):
        """ Retrieve information related to a query. """
        retrieved_docs = await vs.asimilarity_search(state["question"])
        serialized = "\n\n".join((f"Source: {doc.metadata}\n" f"Content: {doc.page_content}") for doc in retrieved_docs)
        return serialized, retrieved_docs
    return retrieve

async def make_generate(llm: BaseChatModel):
    async def generate(state: MessagesState):
        """Generate answer."""
        recent_tool_messages = []
        for message in reversed(state['messages']):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state['messages']
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages
        response = await llm.ainvoke(prompt)
        return { "messages": [response] }
    return generate

async def make_query_or_respond(llm: BaseChatModel, vector_store: VectorStore, trimmer):
    async def query_or_respond(state: MessagesState):
        """Generate tool call for retrieval or respond."""
        trimmed_messages = await trimmer.ainvoke(state['messages'])
        llm_with_tools = llm.bind_tools([await make_retrieve(vector_store)])
        response = await llm_with_tools.ainvoke(trimmed_messages)
        return { "messages": [response] }
    return query_or_respond

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default=os.environ.get("RAG_URL"), help="The URL to scrape for context")
    parser.add_argument("--max-pages", type=int, default=500)
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--memory-tokens", type=int, default=65, help="The more memory the more the model remembers")
    parser.add_argument("--model", "-m", type=str, default="llama3.2", choices=["mistral-nemo", "llama3.2", "gpt-4o-mini"])
    parser.add_argument("--provider", "-p", type=str, default="ollama", choices=["ollama", "openai"])
    args = parser.parse_args()

    if args.url is None:
        raise ValueError("Please provide a URL to scrape")

    llm = init_chat_model(model=args.model, model_provider=args.provider)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large") if args.provider == "openai" else OllamaEmbeddings(model=args.model)
    memory = MemorySaver()
    # TODO: Switch to PGVector
    vector_store = InMemoryVectorStore(embeddings)
    all_splits = await load_resources(args.url, args.max_pages, args.debug)
    _ = await vector_store.aadd_documents(documents=all_splits)

    retrieve = await make_retrieve(vector_store)

    agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)

    png_data = agent_executor.get_graph().draw_mermaid_png()
    with open ("diagram-rag-agent.png", "wb") as f:
        f.write(png_data)

    config = RunnableConfig(configurable={ "thread_id": "abc123" })

    messages = []

    while True:
        user_input = input("\n\nYou: ")
        if user_input.lower() == "quit":
            break

        input_messages = messages + [HumanMessage(content=user_input)]
        print("\n\nRobot: ", end="", flush=True)
        assistent_response = ""

        async for chunk,metadata in agent_executor.astream(
            {"messages": input_messages},
            stream_mode="messages",
            config=config
        ):
            if isinstance(chunk, AIMessage):
                print(chunk.content, end="", flush=True)
                if isinstance(chunk.content, str):
                    assistent_response += chunk.content
            messages.extend([
                HumanMessage(content=user_input),
                SystemMessage(content=assistent_response),
            ])


if __name__ == "__main__":
    asyncio.run(main())
