import pickle
import networkx as nx
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory="vector_store", embedding_function=embeddings)

def load_graph():
    with open("graph_store/knowledge_graph.gpickle", "rb") as f:
        return pickle.load(f)

def vector_retrieve(query, vectordb, k=4):
    results = vectordb.similarity_search(query, k=k)
    return [r.page_content for r in results]

def graph_retrieve(query, G, hops=2):
    query_lower = query.lower()
    matched_nodes = [n for n in G.nodes if n.lower() in query_lower]
    facts = []
    for node in matched_nodes:
        lengths = nx.single_source_shortest_path_length(G, node, cutoff=hops)
        for target, dist in lengths.items():
            if target != node and G.has_edge(node, target):
                rel = G[node][target].get("relation", "related_to")
                facts.append(f"{node} --{rel}--> {target}")
    return facts

def hybrid_retrieve(query, vectordb, G, k=4):
    vec_chunks = vector_retrieve(query, vectordb, k)
    graph_facts = graph_retrieve(query, G)
    return vec_chunks, graph_facts

if __name__ == "__main__":
    vectordb = load_vector_store()
    G = load_graph()
    test_query = input("Enter a test query: ")
    vec_chunks, graph_facts = hybrid_retrieve(test_query, vectordb, G)

    print("\n--- VECTOR CHUNKS ---")
    for c in vec_chunks:
        print(c[:200], "\n---")

    print("\n--- GRAPH FACTS ---")
    for f in graph_facts:
        print(f)