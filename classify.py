import argparse
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

tagging_prompt = ChatPromptTemplate.from_template(
    """
Your a lingustic expert and you have been tasked with classifying a piece of text.

Extract the desired information from the following passage.

It's really really imporant that you follow the properties of the
'Classification' function exactly.

Make sure to think twice before making your decisions. And always take an extra
look when you encounter new languages that might not be as familar to you.

Passage:
{input}
    """
)

class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text, one of 'positive', 'negative', or 'neutral'")
    aggressiveness: int = Field(description="How aggressive the text is on a scale from 1 to 10, where 1 is not aggressive and 10 is very aggressive")
    language: str = Field(description="The language the text is written in")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="The input text to classify")
    args = parser.parse_args()

    print(f"Input text: {args.input}")

    llm = ChatOllama(temperature=0.0, model="llama3.2").with_structured_output(Classification)
    prompt = tagging_prompt.invoke({ "input": args.input })
    response = llm.invoke(prompt)
    print(response)


if __name__ == "__main__":
    main()
