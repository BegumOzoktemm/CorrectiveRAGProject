from typing import Any, Dict
from graph.chain.retrieval_grader import retrival_grader
from graph.state import GraphState

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant,we will set a flag to run web search

    Args:
        state (dict): The current state of Graph

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    print("---CHECK DOCUMENT RELEVANT TO QUESTION")

    question = state["question"]
    documents = state["documents"]

    web_search = False
    filtered_docs=[]
    for d in documents:
        score = retrival_grader.invoke(
            {"question": question, "document": d.page_content}
        )

        grade = score.binary_score

        if grade.lower() =="yes":
            print("---- GRADE: DOCUMENT RELEVANT")
            filtered_docs.append(d)
        else:
            print("---- GRADE: DOCUMENT NOT RELEVANT")
            web_search = True
            continue
    return {"question": question, "documents": filtered_docs, "web_search": web_search}