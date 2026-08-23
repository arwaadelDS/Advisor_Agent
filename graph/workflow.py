from langgraph.graph import StateGraph, START, END
from graph.state import AdvisorState
from agents.supervisor import supervisor_node
from agents.search_agent import search_agent_node
# from agents.rag_agent import rag_agent_node
# from agents.sql_agent import sql_agent_node

graph = StateGraph(AdvisorState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("search_agent", search_agent_node)
# graph.add_node("rag_agent", rag_agent_node)
# graph.add_node("sql_agent", sql_agent_node)

graph.add_edge(START, "supervisor")

def route_next(state: AdvisorState) -> str:
    return state.get("next_agent", "end")

graph.add_conditional_edges(
    "supervisor",
    route_next,
    {
        "sql_agent": END,
        "rag_agent": END,
        "search_agent": "search_agent",
        "end": END,
    }
)

graph.add_edge("search_agent", END)
# graph.add_edge("rag_agent", END)
# graph.add_edge("sql_agent", END)

app = graph.compile()
