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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import List, TypedDict


async def load_resources():
    url = os.environ.get("RAG_URL")
    if url is None:
        return []
    urls = await collect_urls.afrom_website(
        start_url=url,
        url_limit=1000,
        concurrency=10,
        pattern=None
    )

    print(f"Loading resources from {urls}")
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

async def make_query_or_respond(llm: BaseChatModel, vector_store: VectorStore):
    async def query_or_respond(state: MessagesState):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = llm.bind_tools([await make_retrieve(vector_store)])
        response = await llm_with_tools.ainvoke(state['messages'])
        return { "messages": [response] }
    return query_or_respond

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="llama3.2", choices=["mistral-nemo", "llama3.2", "gpt-4o-mini"])
    parser.add_argument("--provider", "-p", type=str, default="ollama", choices=["ollama", "openai"])
    args = parser.parse_args()

    llm = init_chat_model(model=args.model, model_provider=args.provider)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large") if args.provider == "openai" else OllamaEmbeddings(model=args.model)
    memory = MemorySaver()
    # TODO: Switch to PGVector
    vector_store = InMemoryVectorStore(embeddings)
    all_splits = await load_resources()
    _ = await vector_store.aadd_documents(documents=all_splits)

    retrieve = await make_retrieve(vector_store)
    query_or_respond = await make_query_or_respond(llm, vector_store)
    tools = ToolNode([retrieve])
    generate = await make_generate(llm)

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges("query_or_respond", tools_condition, { END: END, "tools": "tools" })
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    graph = graph_builder.compile(checkpointer=memory)

    png_data = graph.get_graph().draw_mermaid_png()
    with open("diagram-rag.png", "wb") as f:
        f.write(png_data)

    config = RunnableConfig(configurable={ "thread_id": "abc123" })

    messages = []

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "quit":
            break

        input_messages = messages + [HumanMessage(content=user_input)]
        print("Robot: ", end="", flush=True)
        assistent_response = ""

        async for chunk,metadata in graph.astream(
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
