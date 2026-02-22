from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import router_node, chat_node, search_node, pdf_node


def route_decision(state: AgentState) -> str:
    """Conditional edge: reads next_action set by the router node."""
    return state.get("next_action", "chat")


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # --- Register nodes ---
    graph.add_node("router", router_node)
    graph.add_node("chat", chat_node)
    graph.add_node("search", search_node)
    graph.add_node("pdf", pdf_node)

    # --- Entry point ---
    graph.set_entry_point("router")

    # --- Conditional edges from router ---
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "chat":   "chat",
            "search": "search",
            "pdf":    "pdf",
        },
    )

    # --- All action nodes go to END after responding ---
    graph.add_edge("chat",   END)
    graph.add_edge("search", END)
    graph.add_edge("pdf",    END)

    return graph.compile()


# Compiled graph — imported by main.py
agent = build_graph()
