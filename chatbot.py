from typing import Sequence
from typing_extensions import Annotated,TypedDict
import asyncio
import argparse
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, trim_messages
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

    trimmer = trim_messages(
        max_tokens=65,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You're a grumpy old southern state paitriot who always thinks it was better in the old days."),
        MessagesPlaceholder(variable_name="messages")
    ])

    async def call_model(state):
        trimmed_messages = await trimmer.ainvoke(state["messages"])
        prompt = prompt_template.invoke({ "messages": trimmed_messages, "language": state["language"] })
        response = await model.ainvoke(prompt)
        return { "messages": response }

    workflow = StateGraph(state_schema=State)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    language = "Swedish"

    messages = [
        SystemMessage(content="you're a good assistant"),
        HumanMessage(content="hi! I'm bob"),
        AIMessage(content="hi!"),
        HumanMessage(content="I like vanilla ice cream"),
        AIMessage(content="nice"),
        HumanMessage(content="whats 2 + 2"),
        AIMessage(content="4"),
        HumanMessage(content="thanks"),
        AIMessage(content="no problem!"),
        HumanMessage(content="having fun?"),
        AIMessage(content="yes!"),
    ]

    config = RunnableConfig(configurable={ "thread_id": "abc123" })
    query = "What math problem did I ask?"
    input_messages = messages + [HumanMessage(query)]

    async for chunk, metadata in app.astream({ "messages": input_messages, "language": language }, config, stream_mode="messages"):
        if isinstance(chunk, AIMessage):
            print(chunk.content, end="", flush=True)
        
if __name__ == "__main__":
    asyncio.run(main())
