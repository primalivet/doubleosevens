import argparse
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.utils.function_calling import tool_example_to_messages

class Person(BaseModel):
    """Information about a person."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.
    
    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(default=None, description="The color of the persons hair")
    height_in_meters: Optional[float] = Field(default=None, description="The height of the person in meters")

class People(BaseModel):
    """Extracted data about people."""
    people: List[Person]

examples = [
    (
        "The ocean is vast and blue. It's more than 20,000 feet deep.",
        People(people=[]),
    ),
    (
        "Fiona traveled far from France to Spain.",
        People(people=[Person(name="Fiona", height_in_meters=None, hair_color=None)]),
    ),
]

message_no_extraction = {
    "role": "user",
    "content": "The solar system is large, but earth has only 1 moon.",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-examples", action="store_true")
    args = parser.parse_args()

    llm = init_chat_model(model="llama3.2", model_provider="ollama")
    structured_llm  = llm.with_structured_output(schema=People)
    messages = []
    for txt, tool_call in examples:
        if tool_call.people:
            ai_response = "Detected people"
        else:
            ai_response = "No people detected"
        messages.extend(tool_example_to_messages(txt, [tool_call], ai_response=ai_response))
    for message in messages:
        message.pretty_print()
    if args.with_examples:
        result = structured_llm.invoke(messages + [message_no_extraction])
        print(result)
    else:
        result = structured_llm.invoke([message_no_extraction])
        print(result)


if __name__ == "__main__":
    main()
