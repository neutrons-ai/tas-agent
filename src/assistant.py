import difflib
import re

from langchain_core.messages import SystemMessage, HumanMessage

### Q&A chatbot — answer general triple-axis questions from the knowledge base

QA_SYSTEM_PROMPT = """You are a neutron scattering assistant for Triple Axis Spectrometers \
(TAS). You answer two kinds of questions from the provided context:
1. Physics/method questions: instrument geometry, resolution functions \
(Cooper-Nathans, Popovici), reciprocal-space scans, and data collection.
2. Instrument facts: publications, staff/team members, capabilities, and \
sample environments for a given instrument.

Rules:
- Answer using ONLY the information in the provided context.
- If the context does not contain the answer, say so plainly instead of guessing.
- For listing questions (publications, staff, etc.), extract the relevant items \
from the context and present them as a list. Do NOT claim the context lacks a \
list if individual items (e.g. paper titles, author names) are present — list \
those items even if a summary count is also shown.
- For physics questions, be concise and technical; use equations or variable \
names exactly as they appear in the context.
- When helpful, cite the source using the `source_file` shown in the context."""


# Enumeration/listing questions ("list all publications", "who is on the team")
# map cleanly to a single source page. Similarity search alone handles these
# poorly because the answer is spread across many chunks and no single chunk
# resembles the query, so we route them to a source-scoped, high-top_k lookup.
PAGE_INTENTS = {
    # The instrument's headline specs (monochromator, flux, momentum range, …)
    # live in a "Specifications" section on its main landing page, NOT on the
    # "/capabilities" sub-page, so this intent routes to the landing page (see
    # LANDING_PAGE_INTENTS). Keep it before "capabilities" so "specification"
    # matches here rather than there.
    "specifications": ["specification", "spec sheet", "specs"],
    "publications": ["publication", "paper", "cite", "reference", "bibliograph"],
    "team": ["team", "staff", "who work", "people", "member", "scientist"],
    "users": ["user", "proposal", "beam time", "beamtime", "apply to use"],
    "capabilities": ["capabilit", "configuration"],
    "sample": ["sample environment", "cryostat", "furnace", "dilution"],
}
# Intents served by the instrument's main landing page (the bare
# "/<instrument>" URL) rather than a "/<instrument>/<page>" sub-page.
LANDING_PAGE_INTENTS = {"specifications"}
# The knowledge base spans several instruments, so a page-targeted question
# must also identify WHICH instrument. Map each instrument's URL path segment
# to the words a user might use for it (name + HFIR beamline code).
INSTRUMENT_ALIASES = {
    "veritas": ["veritas", "hb1a", "hb-1a"],
    "ptax": ["ptax", "hb1", "hb-1"],
    "tax": ["tax", "hb3", "hb-3"],
    "ctax": ["ctax", "cg4c", "cg-4c"],
}
SCOPED_TOP_K = 30  # pull a large slice of the target page for enumeration


def _match_instrument(q):
    """Return the instrument segment the question refers to, or None.

    First try exact word-boundary alias matches (so "tax" doesn't match inside
    "ptax"/"ctax"). If none hit, fall back to a fuzzy pass so common
    misspellings (e.g. "vertias" → "veritas") still route correctly. The fuzzy
    pass only considers name-like aliases (>=4 chars) so short beamline codes
    such as "hb1"/"hb3" can't fuzzily collide with one another.
    """
    for seg, aliases in INSTRUMENT_ALIASES.items():
        if any(re.search(r"\b" + re.escape(a) + r"\b", q) for a in aliases):
            return seg
    tokens = re.findall(r"[a-z0-9-]+", q)
    for seg, aliases in INSTRUMENT_ALIASES.items():
        names = [a for a in aliases if len(a) >= 4]
        if any(difflib.get_close_matches(t, names, n=1, cutoff=0.8) for t in tokens):
            return seg
    return None


def _route_to_source(question, retriever):
    """If the question targets a specific instrument's web page, return that
    page's ``source_file`` value so retrieval can be scoped to it; else None."""
    q = question.lower()
    page = next(
        (p for p, kws in PAGE_INTENTS.items() if any(kw in q for kw in kws)), None
    )
    if not page:
        return None
    instrument = _match_instrument(q)
    if not instrument:
        return None
    try:
        got = retriever.vector_store.collection.get(
            where={"file_type": "web"}, include=["metadatas"]
        )
    except Exception:
        return None
    sources = {m.get("source_file", "") for m in got.get("metadatas", [])}
    # Match the source for this instrument. Landing-page intents resolve to the
    # bare "/<instrument>" URL; sub-page intents match the source whose path
    # contains both the instrument segment and the page (substring, since e.g.
    # ptax uses "/sample-environments" for sample).
    for s in sources:
        path = s.rstrip("/")
        if page in LANDING_PAGE_INTENTS:
            if path.endswith("/" + instrument):
                return s
        elif ("/" + instrument) in path and ("/" + page) in path:
            return s
    return None


def chat(question, retriever, llm, top_k=5):
    """Answer a natural-language question about triple-axis spectrometers using retrieved context."""
    # Route page-targeted / enumeration questions to a source-scoped lookup;
    # otherwise fall back to ordinary similarity search across the whole store.
    source = _route_to_source(question, retriever)
    if source:
        results = retriever.retrieve(
            question, top_k=max(top_k, SCOPED_TOP_K), where={"source_file": source}
        )
    else:
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