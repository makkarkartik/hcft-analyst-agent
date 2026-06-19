from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pymongo import MongoClient

from hcft_agent.config import settings


@tool
def count_chunks_by_state(state: str) -> int:
    """Count report chunks for a 2-letter US state code (e.g. 'CA'). Use for 'how many' questions."""
    coll = MongoClient(settings.mongo_uri)[settings.mongo_db][settings.chunks_collection]
    return coll.count_documents({"state": state.upper()})


llm = ChatOpenAI(
    base_url=settings.orchestrator_base_url,
    api_key=settings.orchestrator_api_key,
    model=settings.orchestrator_model,
    temperature=0.0,
)
llm_with_tools = llm.bind_tools([count_chunks_by_state])


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def agent(state: AgentState) -> dict:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


builder = StateGraph(AgentState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode([count_chunks_by_state]))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()


if __name__ == "__main__":
    # PREDICT: how many times does the agent node fire, and why?
    result = graph.invoke(
        {"messages": [HumanMessage("How many report chunks do we have for California?")]}
    )
    for msg in result["messages"]:
        msg.pretty_print()