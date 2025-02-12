from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

model = ChatOllama(model="deepseek-r1:7b")

def basic_invoke():
    messages = [
        SystemMessage("Translate the following from English to Swedish"),
        HumanMessage("Hey man, how are you doing today? All good?"),
    ]
    model.invoke(messages)
    for token in model.stream(messages):
        print(token.content, end="")

def prompt_template_invoke():
    system_template = "Translate the following from English to {language}"
    prompt_template = ChatPromptTemplate([("system", system_template), ("user", "{text}")])
    prompt = prompt_template.invoke({ "language": "Swedish", "text": "Hallå, hur kom du hit?" })
    model.invoke(prompt)
    for token in model.stream(prompt):
        print(token.content, end="")

if __name__ == "__main__":
    prompt_template_invoke()
