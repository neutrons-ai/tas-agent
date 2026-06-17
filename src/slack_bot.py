### Slack bot wrapper around the RAG pipeline (Socket Mode — no public URL needed)
import os
import sys
from pathlib import Path

# Make the repo root importable so `data.url.url_list` resolves regardless of
# the working directory the bot is launched from.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Load SLACK_BOT_TOKEN / SLACK_APP_TOKEN from a .env file (see .env.example)
load_dotenv()
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from data_loader import process_all_txts, process_all_pdfs, process_all_urls
from chunking import split_documents
from embedding import EmbeddingManager
from vector_store import VectorStore
from rag_retriever import RAGRetriever
from assistant import invoke
from data.url.url_list import url_list


def build_retriever(data_dir=None):
    """Load docs, embed them, and return a ready-to-query retriever. Runs ONCE at startup."""
    # Default to <repo>/data so the path is correct from any launch directory.
    if data_dir is None:
        data_dir = str(REPO_ROOT / "data")
    documents = (
        process_all_pdfs(data_dir)
        + process_all_txts(data_dir)
        + process_all_urls(url_list)
    )
    chunks = split_documents(documents)

    embedding_manager = EmbeddingManager()
    vector_store = VectorStore(
        persistent_directory=str(REPO_ROOT / "data" / "vector_store")
    )

    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)
    vector_store.add_documents(chunks, embeddings)

    return RAGRetriever(vector_store=vector_store, embedding_manager=embedding_manager)


# --- Build the heavy pipeline a single time, before the bot starts listening ---
print("Building RAG index...")
retriever = build_retriever()
llm = ChatOllama(model="gemma4:latest", temperature=0.1, max_completion_tokens=1024)
print("Index ready.")

app = App(token=os.environ["SLACK_BOT_TOKEN"])


def answer(text, mode):
    """Run the request through the RAG pipeline in the given mode."""
    return invoke(request=text.strip(), retriever=retriever, llm=llm, top_k=5, mode=mode)


def run_command(command, ack, respond, mode):
    """Shared slash-command handler: ack within Slack's 3s deadline, then post the answer."""
    ack()  # must happen within 3s; the LLM call below takes longer
    text = command.get("text", "").strip()
    if not text:
        respond(f"Usage: `/{command['command'].lstrip('/')} <your request>`")
        return
    respond(answer(text, mode))


@app.command("/chat")
def handle_chat(command, ack, respond):
    """Answer a triple-axis question from the knowledge base."""
    run_command(command, ack, respond, mode="chat")


@app.command("/scan")
def handle_scan(command, ack, respond):
    """Generate triple-axis scan command(s) from a natural-language request."""
    run_command(command, ack, respond, mode="assistant")


if __name__ == "__main__":
    # App-level token (xapp-...) is what Socket Mode connects with
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
