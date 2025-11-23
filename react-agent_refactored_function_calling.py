from dotenv import load_dotenv
from typing import List
import os
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from callback import AgentCallbackHandler

load_dotenv()


@tool
def get_len_of_text(text: str) -> int:
    """Function to get the length of a string"""
    text = text.strip("'\n").strip('"')
    return len(text)


# Define tools
tools = [get_len_of_text]

# Initialize LLM with function calling support
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    callbacks=[AgentCallbackHandler()],
)

# Bind tools to the model
llm_with_tools = llm.bind_tools(tools)


def run_agent(query: str):
    """Run the agent with function calling"""
    messages = [HumanMessage(content=query)]
    
    print(f"Initial Query: {query}\n")
    
    # Agent loop
    max_iterations = 10
    for i in range(max_iterations):
        print(f"--- Iteration {i + 1} ---")
        
        # Call LLM
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # Check if tool calls are present
        if not response.tool_calls:
            # No tool calls, agent is done
            print(f"\nFinal Answer: {response.content}")
            return response.content
        
        # Execute tool calls
        for tool_call in response.tool_calls:
            print(f"Tool Call: {tool_call['name']}")
            print(f"Tool Input: {tool_call['args']}")
            
            # Find and execute the tool
            selected_tool = None
            for t in tools:
                if t.name == tool_call['name']:
                    selected_tool = t
                    break
            
            if selected_tool:
                tool_output = selected_tool.invoke(tool_call['args'])
                print(f"Tool Output: {tool_output}\n")
                
                # Add tool response to messages
                messages.append(
                    ToolMessage(
                        content=str(tool_output),
                        tool_call_id=tool_call['id']
                    )
                )
            else:
                print(f"Tool {tool_call['name']} not found\n")
                messages.append(
                    ToolMessage(
                        content=f"Error: Tool {tool_call['name']} not found",
                        tool_call_id=tool_call['id']
                    )
                )
    
    print("Max iterations reached")
    return None


if __name__ == "__main__":
    run_agent("How many letters are in the word 'hello'?")

