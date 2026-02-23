from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from agent.state import AgentState
from agent.nodes import router_node, chat_node, search_node, pdf_node


def route_decision(state: AgentState) -> str:
    """Conditional edge: reads next_action set by the router node."""
    return state.get("next_action", "chat")


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("chat",   chat_node)
    graph.add_node("search", search_node)
    graph.add_node("pdf",    pdf_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_decision,
        {"chat": "chat", "search": "search", "pdf": "pdf"},
    )

    graph.add_edge("chat",   END)
    graph.add_edge("search", END)
    graph.add_edge("pdf",    END)

    # Attach checkpointer if provided — enables persistent memory per thread_id
    return graph.compile(checkpointer=checkpointer)
