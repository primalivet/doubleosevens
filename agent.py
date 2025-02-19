from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableConfig

def main():
    memory = MemorySaver()
    model = ChatOllama(model="llama3.2")
    search = TavilySearchResults(max_results=2)
    tools = [search]
    agent_executor = create_react_agent(model, tools, checkpointer=memory)

    config = RunnableConfig(configurable= {"thread_id": "abc123"})
    for chunk in agent_executor.stream({ "messages": [HumanMessage(content="hi im bob! and i live in sf")] }, config):
        print(chunk)
        print("----")

    for chunk in agent_executor.stream({ "messages": [HumanMessage(content="whats the weather where i live?")] }, config):
        print(chunk)
        print("----")

if __name__ == "__main__":
    main()
