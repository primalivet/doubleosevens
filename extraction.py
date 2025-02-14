import os
import argparse
from typing import Optional, List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("description", type=str)
    parser.add_argument("--model", "-m", type=str, default="llama3.2", choices=["mistral-nemo", "llama3.2", "gpt-4o-mini"])
    parser.add_argument("--provider", "-p", type=str, default="ollama", choices=["ollama", "openai"])
    args = parser.parse_args()

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value."),
        ("human", "{text}")
    ])
    llm = init_chat_model(model=args.model, model_provider=args.provider)
    structured_llm = llm.with_structured_output(schema=People)
    prompt = prompt_template.invoke({"text": args.description})
    result = structured_llm.invoke(prompt)
    print(result)


if __name__ == "__main__":
    main()
