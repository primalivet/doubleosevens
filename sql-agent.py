import asyncio
import re
import ast
from langchain import hub
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent

def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))

async def main():
    db = SQLDatabase.from_uri('sqlite:///dbs/Chinook.db')
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = InMemoryVectorStore(embeddings)

    llm = init_chat_model(model="gpt-4o-mini", model_provider="openai")
    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
    assert len(prompt_template.messages) == 1

    prompt_template.pretty_print()

    artists = query_as_list(db, "SELECT Name FROM Artist")
    albums = query_as_list(db, "SELECT Title FROM Album")
    _ = vector_store.add_texts(artists + albums)
    retriver = vector_store.as_retriever(search_kwargs={"top_k": 5})
    description = (
        "Use to look up values to filter on. Input is an approximate spelling "
        "of the proper noun, output is valid proper nouns. Use the noun most "
        "similar to the search."
    )
    retriver_tool = create_retriever_tool(
        retriver,
        name="search_proper_nouns",
        description=description,
    )


    suffix = (
        "If you need to filter on a proper noun like a Name, you must ALWAYS first look up "
        "the filter value using the 'search_proper_nouns' tool! Do not try to "
        "guess at the proper name - use this function to find similar ones."
    )

    system_message = prompt_template.format(dialect=db.dialect, top_k=10)
    system = f"{system_message}\n\n{suffix}"
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    tools.append(retriver_tool)
    agent_executor = create_react_agent(llm, tools, state_modifier=system)

    png_data = agent_executor.get_graph().draw_mermaid_png()
    with open ("diagram-sql-agent.png", "wb") as f:
        f.write(png_data)

    question = "Which country's customers spent the most?"

    for step in agent_executor.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

    question = "Describe the playlisttrack table"

    for step in agent_executor.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

    question = "How many albums does alis in chain have?"

    for step in agent_executor.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

if __name__ == '__main__':
    asyncio.run(main())
