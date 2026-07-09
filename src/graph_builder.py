import os, json, pickle, time
import networkx as nx
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

EXTRACTION_PROMPT = """Extract factual (subject, relationship, object) triples from the text below.
Return ONLY valid JSON, a list of objects like:
[{{"subject": "...", "relation": "...", "object": "..."}}]
No preamble, no markdown, no explanation.

Text:
{text}
"""

def safe_str(value):
    """Safely convert any value (including None) to a stripped string."""
    if value is None:
        return ""
    return str(value).strip()

def extract_triples(text, max_retries=3):
    prompt = EXTRACTION_PROMPT.format(text=text)
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt).content
            break
        except Exception as e:
            print(f"API error (attempt {attempt+1}): {e}")
            time.sleep(20)
    else:
        return []

    response = response.strip().replace("```json", "").replace("```", "")
    try:
        triples = json.loads(response)
        if not isinstance(triples, list):
            return []
        return triples
    except json.JSONDecodeError:
        print("Failed to parse:", response[:200])
        return []

def build_graph(chunks, checkpoint_every=20):
    G = nx.DiGraph()
    for i, chunk in enumerate(chunks):
        triples = extract_triples(chunk.page_content)
        for t in triples:
            if not isinstance(t, dict):
                continue
            subj = safe_str(t.get("subject"))
            rel = safe_str(t.get("relation"))
            obj = safe_str(t.get("object"))
            if subj and obj:
                G.add_edge(subj, obj, relation=rel, source_chunk=i)
        print(f"Processed chunk {i+1}/{len(chunks)} — {len(triples)} triples")
        time.sleep(2)

        # Save progress periodically so a crash never loses everything
        if (i + 1) % checkpoint_every == 0:
            save_graph(G)
            print(f"--- Checkpoint saved at chunk {i+1} ---")

    return G

def save_graph(G, path="graph_store/knowledge_graph.gpickle"):
    os.makedirs("graph_store", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)
    print(f"Graph saved: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

if __name__ == "__main__":
    import sys
    sys.path.append("src")
    from ingest import load_documents, chunk_documents
    docs = load_documents()
    chunks = chunk_documents(docs)

    from collections import defaultdict
    chunks_by_source = defaultdict(list)
    for c in chunks:
        source = c.metadata.get("source", "unknown")
        chunks_by_source[source].append(c)

    PER_PDF_LIMIT = 40
    sampled_chunks = []
    for source, source_chunks in chunks_by_source.items():
        sampled_chunks.extend(source_chunks[:PER_PDF_LIMIT])
        print(f"{source}: taking {min(PER_PDF_LIMIT, len(source_chunks))} of {len(source_chunks)} chunks")

    chunks = sampled_chunks
    print(f"Total chunks selected for graph: {len(chunks)}")

    G = build_graph(chunks)
    save_graph(G)