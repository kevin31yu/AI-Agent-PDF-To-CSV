from typing import Annotated, Literal
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # add_messages is a reducer: new messages are appended, not overwritten
    messages: Annotated[list, add_messages]

    # The router sets this to tell the graph which path to take
    next_action: Literal["chat", "search", "pdf"] | None

    # Only used when next_action == "pdf"
    pdf_path: str | None
    csv_path: str | None
