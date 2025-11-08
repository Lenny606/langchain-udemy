from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    information = """"
    Albert Einstein was born in Berlin, Germany. Famous scients.
    """

    template = """
    given info in {information}
    - return short biography
    - + return 2 fun facts
    """

    prompt_template = PromptTemplate(
        input_variables=["information"],
        template=template
    )

    # llm = ChatOpenAI(
    #     temperature=0.5,
    #     model="gpt-4o-mini"
    # )

    # ollama
    llm = ChatOllama(
        temperature=0.5,
        model="gemma3:270m"
    )
    #     chaining in LCEL
    chain = prompt_template | llm
    result = chain.invoke(
        input={"information": information}
    )

    print(result.content)


if __name__ == "__main__":
    main()
