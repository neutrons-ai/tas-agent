from langchain_core.messages import SystemMessage, HumanMessage

### Q&A chatbot — answer general triple-axis questions from the knowledge base

QA_SYSTEM_PROMPT = """You are a neutron scattering assistant that answers questions about \
Triple Axis Spectrometers (TAS): instrument geometry, resolution functions \
(Cooper-Nathans, Popovici), reciprocal-space scans, and data collection.

Rules:
- Answer using ONLY the information in the provided context.
- If the context does not contain the answer, say so plainly instead of guessing.
- Be concise and technical; use equations or variable names exactly as they appear in the context.
- When helpful, cite the source paper using the `source_file` shown in the context."""


def chat(question, retriever, llm, top_k=5):
    """Answer a natural-language question about triple-axis spectrometers using retrieved context."""
    # Retrieve relevant passages from the knowledge base
    results = retriever.retrieve(question, top_k=top_k)
    if not results:
        return "No relevant information found in the knowledge base."

    # Build context with source attribution so the model can cite papers
    context = "\n\n".join(
        f"[Source: {doc['metadata'].get('source_file', 'unknown')}]\n{doc['content']}"
        for doc in results
    )

    response = llm.invoke([
        SystemMessage(QA_SYSTEM_PROMPT),
        HumanMessage(f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ])
    return response.content.strip()

### Generate triple-axis scan commands from a natural-language request

SCAN_SYSTEM_PROMPT = """You are an assistant that converts natural-language requests \
into Triple Axis instrument scan commands.

Rules:
- Use ONLY the command syntax shown in the provided context.
- Output ONLY the command(s), one per line. No explanation, no markdown, no code fences.
- If a monitor count (preset) or scan title is requested, emit those commands first.
- For a reciprocal-space scan use: scan h <start> <stop> <step> k <k> l <l> e <e>
- If the request is missing a value, use 0 for that coordinate.
- If the request contain typos, reply the mistake and suggest a fix based on the documented commands."""


def generate_scan_command(request, retriever, llm, top_k=3):
    """Translate a natural-language request into triple-axis scan command(s)."""
    # Retrieve the command syntax from the knowledge base
    results = retriever.retrieve(request, top_k=top_k)
    context = "\n\n".join([doc["content"] for doc in results]) if results else ""

    if not context:
        return "No command syntax found in the knowledge base."

    response = llm.invoke([
        SystemMessage(SCAN_SYSTEM_PROMPT),
        HumanMessage(f"Command reference:\n{context}\n\nRequest: {request}\n\nCommands:"),
    ])
    return response.content.strip()

def invoke(request, retriever, llm, top_k, mode):
    if mode == "chat":
        return chat(question=request, retriever=retriever, llm=llm, top_k=top_k)
    elif mode == "assistant":
        return generate_scan_command(request=request, retriever=retriever, llm=llm, top_k=top_k)