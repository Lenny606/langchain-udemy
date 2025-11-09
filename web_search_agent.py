from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from schemas import AgentResponse

load_dotenv()

# react_prompt = hub.pull("hwchase17/react")

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


def run_agent():
    result = agent.invoke({"messages": [
        {"role": "user",

         "content": "Search the web for 3 jobs in Lisabon in IT field"}]})
    print(result)


if __name__ == "__main__":
    run_agent()
