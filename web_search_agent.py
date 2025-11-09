from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from schemas import AgentResponse
from prompt import REACT_PROMPT_CUSTOM_TEMPLATE
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

load_dotenv()

# react_prompt = hub.pull("hwchase17/react")
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
custom_prompt = PromptTemplate(template=REACT_PROMPT_CUSTOM_TEMPLATE,
                               input_variables=["input", "agent_scratchpad",
                                                "tools", "tool_names"]).partial(
    format_instructions=output_parser.get_format_instructions()
)
# it is Search Tool wrapped as chain
tools = [TavilySearch()]
model = "gpt-4o-mini"
llm = ChatOpenAI(
    temperature=0.5,
    model=model
)

agent = create_agent(
    tools=tools,
    model=llm,
    response_format=AgentResponse,
)

chain = custom_prompt | agent


def run_agent():
    result = agent.invoke({"messages": [
        {"role": "user",

         "content": "Search the web for 3 jobs in Lisabon in IT field"}]})

    # Access structured response from the agent
    structured = result.get("structured_response", None)
    print(structured if structured is not None else result)


if __name__ == "__main__":
    run_agent()
