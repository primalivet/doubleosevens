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
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--model", "-m", type=str, default="llama3.2", choices=["mistral-nemo", "llama3.2", "gpt-4o-mini"])
    parser.add_argument("--provider", "-p", type=str, default="ollama", choices=["ollama", "openai"])
    parser.add_argument("--memory-tokens", type=int, default=65, help="The more memory the more the model remembers")
    args = parser.parse_args()

    if args.debug:
        print(args)

    model = init_chat_model(model=args.model, model_provider=args.provider)

    trimmer = trim_messages(
        max_tokens=args.memory_tokens,
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
        if args.debug:
            print("\n\n Memory: \n")
            print(trimmed_messages)
            print("\n")

        prompt = prompt_template.invoke({ "messages": trimmed_messages, "language": state["language"] })
        response = await model.ainvoke(prompt)
        return { "messages": response }

    workflow = StateGraph(state_schema=State)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    language = "Swedish"

    messages = []

    config = RunnableConfig(configurable={ "thread_id": "abc123" })

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "quit":
            break
        
        input_messages = messages + [HumanMessage(content=user_input)]

        print("\nRobot: ", end="", flush=True)
        assistant_response = ""
        async for chunk, metadata in app.astream({ "messages": input_messages, "language": language }, config, stream_mode="messages"):
            if isinstance(chunk, AIMessage):
                print(chunk.content, end="", flush=True)
                if isinstance(chunk.content, str):
                    assistant_response += chunk.content

        messages.extend([
            HumanMessage(content=user_input),
            AIMessage(content=assistant_response)
        ])

if __name__ == "__main__":
    asyncio.run(main())
