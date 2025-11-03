from typing import Any, Dict
from graph.state import GraphState
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document

web_search_tool = TavilySearchResults(k=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("----WEB SEARCH----")

    question = state["question"]
    documents = state["documents"]

    docs = web_search_tool.invoke({"query": question})
    web_result = "\n".join([d["content"] for d in docs])
    web_result = Document(page_content=web_result)

    if documents is not None:
        documents.append(web_result)
    else:
        documents = [web_result]

    return{"question": question, "documents": documents}