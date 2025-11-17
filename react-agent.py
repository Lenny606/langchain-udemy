from dotenv import load_dotenv
from typing import List, Union
import os
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents.output_parsers.react_single_input import (
    ReActSingleInputOutputParser,
)

from langchain_classic.schema import AgentAction, AgentFinish
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_classic.tools import Tool
from langchain_core.tools.render import render_text_description
from langchain_classic import hub
from langchain.agents import create_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

load_dotenv()

o = hub.pull("hwchase17/react")
# template is from hwcase
template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:"""


@tool
def get_len_of_text(text: str) -> int:
    """Function to get the length of a string"""
    text = text.strip("'\n").strip('"')
    return len(text)

# define tools into arr
tools: List[Tool]= [get_len_of_text]

def find_tool_by_name(tools: List[Tool], name: str) -> Tool:
    """Find a tool by name"""
    for tool in tools:
        if tool.name == name:
            return tool
    raise ValueError(f"Tool with name {name} not found")


# build prompt
prompt = PromptTemplate(
    template=template
).partial(
    # render_text_description converts the tools into a human-readable format
    # tool_names creates a comma-separated string of available tool names
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools])
)
"""
Creates a partially filled PromptTemplate for ReAct agent interaction.
Parameters:
    - tools: List of available tools rendered in human-readable format
    - tool_names: Comma-separated string of tool names available to the agent
Returns:
    A PromptTemplate with tools and tool_names parameters pre-filled
"""

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    stop="Observation"  # stops llm generation when observation token is found, can be diff for every model
)

chain = {
            "input": lambda x: x["input"]
        } | prompt | llm | ReActSingleInputOutputParser()

chain_step: Union[AgentAction, AgentFinish] = chain.invoke({"input": "How many letters are in the word 'hello'?"})

if isinstance(chain_step, AgentAction):
    print(f"Action: {chain_step}")
    tool_name = chain_step.tool
    tool_to_use = find_tool_by_name(tools, tool_name)
    tool_input = chain_step.tool_input

    observation = tool_to_use.func(tool_input)
    print(f"Observation: {observation}")
print(chain_step)
