import os
import re
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent.graph import build_graph
from agent.db import init_db, get_checkpointer, upsert_session, log_conversion, list_sessions

load_dotenv()

BANNER = """
╔══════════════════════════════════════════════╗
║           AI Tax & Research Agent            ║
║  Chat · Web Search · PDF → CSV Converter     ║
╠══════════════════════════════════════════════╣
║  Commands:                                   ║
║    convert <path/to/file.pdf>  — PDF to CSV  ║
║    history                     — past PDFs   ║
║    exit / quit                 — quit        ║
╚══════════════════════════════════════════════╝
"""


def parse_pdf_path(user_input: str) -> tuple[str, str | None]:
    match = re.match(r"convert\s+(.+)", user_input, re.IGNORECASE)
    if match:
        path = match.group(1).strip().strip('"').strip("'")
        return f"convert {path}", path
    return user_input, None


def pick_session() -> str:
    """Prompt the user to start a new session or resume an existing one."""
    sessions = list_sessions()

    if not sessions:
        thread_id = str(uuid.uuid4())
        print(f"  Starting new session: {thread_id[:8]}…\n")
        return thread_id

    print("  Recent sessions:")
    for i, s in enumerate(sessions, 1):
        ts = s["last_active"][:16].replace("T", " ")
        print(f"  [{i}] {s['thread_id'][:8]}…  {ts}  ({s['message_count']} messages)")
    print("  [N] Start new session")
    print()

    choice = input("  Resume session [1-{0}] or N for new: ".format(len(sessions))).strip().upper()

    if choice == "N" or not choice:
        thread_id = str(uuid.uuid4())
        print(f"\n  Starting new session: {thread_id[:8]}…\n")
        return thread_id

    try:
        idx = int(choice) - 1
        thread_id = sessions[idx]["thread_id"]
        print(f"\n  Resuming session: {thread_id[:8]}…\n")
        return thread_id
    except (ValueError, IndexError):
        thread_id = str(uuid.uuid4())
        print(f"\n  Invalid choice — starting new session: {thread_id[:8]}…\n")
        return thread_id


def run():
    print(BANNER)

    # Initialise our custom DB tables
    init_db()

    thread_id = pick_session()
    config = {"configurable": {"thread_id": thread_id}}

    # SqliteSaver used as context manager to ensure the connection is properly closed
    with get_checkpointer() as checkpointer:
        agent = build_graph(checkpointer)

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

            if user_input.lower() == "history":
                from agent.db import list_conversions
                records = list_conversions(thread_id)
                if records:
                    print("\n  PDF conversions in this session:")
                    for r in records:
                        print(f"    {r['processed_at'][:16]}  {r['pdf_file']}  →  {r['csv_file']}")
                else:
                    print("\n  No conversions yet in this session.")
                print()
                continue

            cleaned_input, pdf_path = parse_pdf_path(user_input)

            # Only pass the NEW message — LangGraph restores history from the checkpoint
            state = {
                "messages":   [HumanMessage(content=cleaned_input)],
                "next_action": None,
                "pdf_path":    pdf_path,
                "csv_path":    None,
            }

            result = agent.invoke(state, config=config)

            # The last AIMessage in result is the new reply
            ai_messages = [
                m for m in result["messages"]
                if m.__class__.__name__ == "AIMessage"
            ]

            if ai_messages:
                print(f"\nAgent: {ai_messages[-1].content}\n")

            # Log conversion + update session stats
            if result.get("csv_path"):
                log_conversion(thread_id, pdf_path or "", result["csv_path"])
                print(f"  [CSV saved → {result['csv_path']}]\n")

            # +2 = one user message + one AI reply
            upsert_session(thread_id, delta_messages=2)


if __name__ == "__main__":
    run()
