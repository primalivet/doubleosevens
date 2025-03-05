import asyncio
import argparse
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict, Annotated

class State(TypedDict, total=False):
    question: str
    query: str
    result: str
    answer: str

class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically correct SQL query."]

def make_write_query(db: SQLDatabase, llm):
    """Generate SQL query to fetch information."""
    def write_query(state: State) -> QueryOutput:
        if "question" not in state:
            raise ValueError("No question to generate query for")
        query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
        prompt = query_prompt_template.invoke(
            {
                "dialect": db.dialect,
                "top_k": 10,
                "table_info": db.get_table_info(),
                "input": state["question"],
            }
        )
        structured_llm = llm.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        return { "query": result["query"] }
    return write_query

def make_execute_query(db: SQLDatabase):
    def execute_query(state: State):
        """Execute SQL query and fetch results."""
        execute_query_tool = QuerySQLDatabaseTool(db=db)
        if "query" not in state:
            raise ValueError("No query to exectute")
        return { "result": execute_query_tool.invoke(state["query"]) }
    return execute_query

def make_generate_answer(llm):
    def generate_answer(state: State):
        """Answer question using retrived information as context."""
        if "result" not in state:
            raise ValueError("No result to generate answer from")
        if "question" not in state:
            raise ValueError("No question to generate answer for")
        if "query" not in state:
            raise ValueError("No query to generate answer for")
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the question.\n\n"
            f'Question: {state["question"]}\n'
            f'Query: {state["query"]}\n'
            f'Result: {state["result"]}\n'
        )
        response = llm.invoke(prompt)
        return { "answer": response.content }
    return generate_answer

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="llama3.2", choices=["mistral-nemo", "llama3.2", "gpt-4o-mini"])
    parser.add_argument("--provider", "-p", type=str, default="ollama", choices=["ollama", "openai"])
    args = parser.parse_args()

    db = SQLDatabase.from_uri('sqlite:///dbs/Chinook.db')

    llm = init_chat_model(model=args.model, model_provider=args.provider)

    graph_builder = StateGraph(State).add_sequence(
        [make_write_query(db, llm), make_execute_query(db), make_generate_answer(llm)]
    )
    graph_builder.add_edge(START, "write_query")
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory, interrupt_before=["execute_query"])
    config = RunnableConfig(configurable={"thread_id": "1"})

    png_data = graph.get_graph().draw_mermaid_png()
    with open ("diagram-sql-chain.png", "wb") as f:
        f.write(png_data)

    for step in graph.stream( {"question": "How many employees are there?"}, stream_mode="updates", config=config):
        print(step) # Interupts before execute_query
    
    try:
        state = await graph.aget_state(config)
        query = state.values.get("query")
        if query is None:
            raise ValueError("No query to execute")
        print(f'Robot want\'s to execute datbase query: {query}')
        user_approval = input("Continue? (y/n): ")
    except Exception:
        user_approval = "n"

    if user_approval.lower() == "y":
        for step in graph.stream(None, stream_mode="updates", config=config):
            print(step)
    else:
        print("Denied by user")
        

if __name__ == '__main__':
    asyncio.run(main())
