import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from retriever import load_vector_store, load_graph, hybrid_retrieve

load_dotenv()
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

vectordb = load_vector_store()
G = load_graph()

class RAGState(TypedDict):
    query: str
    original_query: str
    vec_chunks: List[str]
    graph_facts: List[str]
    answer: str
    retries: int
    grounded: bool
    relevant_docs: bool

def retrieve_node(state: RAGState):
    vec_chunks, graph_facts = hybrid_retrieve(state["query"], vectordb, G)
    return {"vec_chunks": vec_chunks, "graph_facts": graph_facts}

def grade_documents_node(state: RAGState):
    context = "\n".join(state["vec_chunks"])
    prompt = f"""Query: {state['query']}
Retrieved context: {context}

Is this context relevant enough to answer the query? Reply with only YES or NO."""
    result = llm.invoke(prompt).content.strip().upper()
    return {"relevant_docs": "YES" in result}

def rewrite_query_node(state: RAGState):
    prompt = f"""The original query "{state['original_query']}" did not retrieve good results.
Rewrite it to be clearer and more specific, using different phrasing. Return ONLY the rewritten query."""
    new_query = llm.invoke(prompt).content.strip()
    return {"query": new_query, "retries": state["retries"] + 1}

def generate_node(state: RAGState):
    context = "\n".join(state["vec_chunks"])
    facts = "\n".join(state["graph_facts"])
    prompt = f"""Answer the question using ONLY the context and graph facts below.
If the answer isn't in there, say "I don't have enough information."

Context:
{context}

Graph facts (relationships between entities):
{facts}

Question: {state['original_query']}

Answer:"""
    answer = llm.invoke(prompt).content.strip()
    return {"answer": answer}

def grade_answer_node(state: RAGState):
    prompt = f"""Context: {state['vec_chunks']}
Answer: {state['answer']}

Is the answer factually grounded in the context (not hallucinated)? Reply only YES or NO."""
    result = llm.invoke(prompt).content.strip().upper()
    return {"grounded": "YES" in result}

def decide_after_grading(state: RAGState):
    if state["relevant_docs"]:
        return "generate"
    elif state["retries"] < 2:
        return "rewrite"
    else:
        return "generate"

def decide_after_answer(state: RAGState):
    if state["grounded"] or state["retries"] >= 2:
        return "end"
    return "regenerate"

workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("rewrite", rewrite_query_node)
workflow.add_node("generate", generate_node)
workflow.add_node("grade_answer", grade_answer_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_after_grading, {
    "generate": "generate",
    "rewrite": "rewrite"
})
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", "grade_answer")
workflow.add_conditional_edges("grade_answer", decide_after_answer, {
    "end": END,
    "regenerate": "generate"
})

app = workflow.compile()

def run_query(query: str):
    initial_state = {
        "query": query,
        "original_query": query,
        "vec_chunks": [],
        "graph_facts": [],
        "answer": "",
        "retries": 0,
        "grounded": False,
        "relevant_docs": False
    }
    result = app.invoke(initial_state)
    return result

if __name__ == "__main__":
    q = input("Ask a question: ")
    result = run_query(q)
    print("\n--- ANSWER ---")
    print(result["answer"])
    print("\n--- GRAPH FACTS USED ---")
    print(result["graph_facts"])
    print(f"\n--- Retries: {result['retries']} | Grounded: {result['grounded']} ---")