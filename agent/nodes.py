import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.state import AgentState
from agent.tools.web_search import search_web
from agent.tools.pdf_converter import convert_pdf_to_csv

load_dotenv()

# ---------------------------------------------------------------------------
# Shared LLM instance
# ---------------------------------------------------------------------------
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0.7,
)

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful AI assistant with three capabilities:\n"
    "1. General conversation and question answering.\n"
    "2. Web search — you can look up current information online.\n"
    "3. PDF to CSV conversion — you can extract tax return data from PDF files.\n\n"
    "Be concise, accurate, and friendly."
))


# ---------------------------------------------------------------------------
# Router node — decides which path to take
# ---------------------------------------------------------------------------
def router_node(state: AgentState) -> AgentState:
    """
    Classify the latest user message into: chat | search | pdf.
    Uses a lightweight LLM call with strict instructions.
    """
    last_message = state["messages"][-1]
    user_text = last_message.content if hasattr(last_message, "content") else str(last_message)

    classification_prompt = [
        SystemMessage(content=(
            "Classify the user's intent into exactly one word — no explanation, no punctuation.\n\n"
            "Rules:\n"
            "- Reply 'pdf'    if they want to convert, process, or extract data from a PDF file.\n"
            "- Reply 'search' if they want current/real-time info, news, prices, or ask you to search the web.\n"
            "- Reply 'chat'   for everything else (general questions, coding help, advice, math, etc.).\n\n"
            "Examples:\n"
            "  'convert my tax pdf' → pdf\n"
            "  'what is the weather today' → search\n"
            "  'explain recursion' → chat\n"
            "  'latest bitcoin price' → search\n"
            "  'process invoice.pdf' → pdf"
        )),
        HumanMessage(content=user_text),
    ]

    response = llm.invoke(classification_prompt)
    action = response.content.strip().lower()

    # Fallback to chat if classification is unexpected
    if action not in ("chat", "search", "pdf"):
        action = "chat"

    return {"next_action": action, "pdf_path": state.get("pdf_path"), "csv_path": state.get("csv_path")}


# ---------------------------------------------------------------------------
# Chat node — general conversation
# ---------------------------------------------------------------------------
def chat_node(state: AgentState) -> AgentState:
    """Standard multi-turn conversation with Claude."""
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)]}


# ---------------------------------------------------------------------------
# Search node — web search + summarise
# ---------------------------------------------------------------------------
def search_node(state: AgentState) -> AgentState:
    """Run a Tavily web search then ask Claude to summarise the results."""
    last_message = state["messages"][-1]
    query = last_message.content if hasattr(last_message, "content") else str(last_message)

    # 1. Fetch results
    raw_results = search_web(query)

    # 2. Ask Claude to summarise them for the user
    summary_prompt = [
        SystemMessage(content=(
            "You are a research assistant. The user asked a question and you ran a web search. "
            "Summarise the search results below into a clear, concise answer. "
            "Cite sources with [1], [2], etc. where relevant."
        )),
        HumanMessage(content=f"User question: {query}\n\nSearch results:\n{raw_results}"),
    ]
    response = llm.invoke(summary_prompt)
    return {"messages": [AIMessage(content=response.content)]}


# ---------------------------------------------------------------------------
# PDF node — convert PDF to CSV
# ---------------------------------------------------------------------------
def pdf_node(state: AgentState) -> AgentState:
    """
    Convert a PDF to a tax-return CSV.
    Expects state['pdf_path'] to be set before this node runs.
    """
    pdf_path = state.get("pdf_path", "").strip()

    if not pdf_path:
        return {
            "messages": [AIMessage(content=(
                "I'd love to convert a PDF for you! Please provide the path to your PDF file.\n"
                "Example: `convert /path/to/tax_document.pdf`"
            ))]
        }

    if not os.path.isfile(pdf_path):
        return {
            "messages": [AIMessage(content=f"File not found: `{pdf_path}`\nPlease check the path and try again.")]
        }

    try:
        csv_path, summary = convert_pdf_to_csv(pdf_path, output_dir="output")

        # Ask Claude to relay the summary in a friendly way
        relay_prompt = [
            SystemMessage(content="You are a helpful assistant. Relay the following PDF conversion result to the user in a clear, friendly way."),
            HumanMessage(content=summary),
        ]
        response = llm.invoke(relay_prompt)
        return {
            "messages": [AIMessage(content=response.content)],
            "csv_path": csv_path,
        }

    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Error processing PDF: {e}")]
        }
