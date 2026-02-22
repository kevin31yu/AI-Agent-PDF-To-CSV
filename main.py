import os
import re
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent.graph import agent

load_dotenv()

BANNER = """
╔══════════════════════════════════════════════╗
║           AI Tax & Research Agent            ║
║  Chat · Web Search · PDF → CSV Converter     ║
╠══════════════════════════════════════════════╣
║  Commands:                                   ║
║    convert <path/to/file.pdf>  — PDF to CSV  ║
║    exit / quit                 — quit         ║
╚══════════════════════════════════════════════╝
"""

# Persistent conversation history across turns
conversation_history: list = []


def parse_pdf_path(user_input: str) -> tuple[str, str | None]:
    """
    If the input contains 'convert <path>', extract the path and
    return a cleaned message + the pdf path.
    Otherwise return the original input + None.
    """
    match = re.match(r"convert\s+(.+)", user_input, re.IGNORECASE)
    if match:
        path = match.group(1).strip().strip('"').strip("'")
        return f"convert {path}", path
    return user_input, None


def run():
    print(BANNER)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Check for PDF conversion command
        cleaned_input, pdf_path = parse_pdf_path(user_input)

        # Append user message to history
        conversation_history.append(HumanMessage(content=cleaned_input))

        # Build state for this turn
        state = {
            "messages": conversation_history,
            "next_action": None,
            "pdf_path": pdf_path,
            "csv_path": None,
        }

        # Run the graph
        result = agent.invoke(state)

        # Extract the latest AI message
        ai_messages = [m for m in result["messages"] if hasattr(m, "content") and m.__class__.__name__ == "AIMessage"]

        if ai_messages:
            reply = ai_messages[-1]
            print(f"\nAgent: {reply.content}\n")
            # Add only the final reply to history (not intermediate router messages)
            conversation_history.append(reply)

        # Show CSV path if one was generated
        if result.get("csv_path"):
            print(f"  [CSV saved → {result['csv_path']}]\n")


if __name__ == "__main__":
    run()
