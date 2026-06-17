### simple rag pipline
import sys
from langchain_ollama import ChatOllama
from data_loader import process_all_txts, process_all_pdfs, process_all_urls

# URL list lives in data/url/url_list.py
sys.path.append("../data/url")
from url_list import url_list
from chunking import split_documents
from embedding import EmbeddingManager
from vector_store import VectorStore
from rag_retriever import RAGRetriever
from assistant import invoke, chat, generate_scan_command

# Process all PDFs, txts, and webpages
all_documents = (
    process_all_pdfs("../data")
    + process_all_txts("../data")
    + process_all_urls(url_list)
)

# chunking

chunks=split_documents(all_documents)

embedding_manager = EmbeddingManager()
vector_store = VectorStore()

### Convert the text to embeddings
texts=[doc.page_content for doc in chunks]

## Generate the Embeddings

embeddings=embedding_manager.generate_embeddings(texts)

##store int he vector dtaabase
vector_store.add_documents(chunks,embeddings)

rag_retriever = RAGRetriever(vector_store=vector_store, embedding_manager= embedding_manager)

llm = ChatOllama(model="gemma4:latest", temperature=0.1, max_completion_tokens=1024)

def select_mode():
    """Ask the user which mode to run until they pick a valid one."""
    while True:
        choice = input("\nMode — [c]hat or [a]ssistant (scan commands)? ").strip().lower()
        if choice in ("c", "chat"):
            return "chat"
        if choice in ("a", "assistant"):
            return "assistant"
        if choice in ("q", "quit", "exit"):
            return None
        print("  Please enter 'c', 'a', or 'q' to quit.")


def main():
    print("Neutron RAG assistant. Type 'q' at any prompt to quit.")
    while True:
        mode = select_mode()
        if mode is None:
            break

        request = input(f"[{mode}] Enter your request: ").strip()
        if request.lower() in ("q", "quit", "exit"):
            break
        if not request:
            print("  Empty request, try again.")
            continue

        answer = invoke(
            request=request,
            retriever=rag_retriever,
            llm=llm,
            top_k=3,
            mode=mode,
        )
        print(f"\n{answer}")

    print("\nGoodbye.")


if __name__ == "__main__":
    main()