from typing import Sequence
from typing_extensions import Annotated,TypedDict
import asyncio
import argparse
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="llama3.2", choices=["mistral-nemo", "llama3.2", "gpt-4o-mini"])
    parser.add_argument("--provider", "-p", type=str, default="ollama", choices=["ollama", "openai"])
    args = parser.parse_args()

    model = init_chat_model(model=args.model, model_provider=args.provider)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You talk like a pirate, Answer all questions to the best of your ability in {language} with a angry pirate tone, but keep it short!"),
        MessagesPlaceholder(variable_name="messages")
    ])

    async def call_model(state):
        prompt = prompt_template.invoke(state)
        response = await model.ainvoke(prompt)
        return { "messages": response }

    workflow = StateGraph(state_schema=State)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    language = "Swedish"

    config = RunnableConfig(configurable={ "thread_id": "abc123" })
    query = "Holla! I'm Bob."
    input_messages = [HumanMessage(query)]
    output = await app.ainvoke({ "messages": input_messages, "language": language }, config)
    output["messages"][-1].pretty_print()

    config = RunnableConfig(configurable={ "thread_id": "abc123" })
    query = "What's my name?"
    input_messages = [HumanMessage(query)]
    output = await app.ainvoke({"messages": input_messages, "language": language }, config)
    output["messages"][-1].pretty_print()
    

if __name__ == "__main__":
    asyncio.run(main())
